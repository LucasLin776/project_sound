"""Run local PANNs feature/tagging/SED diagnostics on an audio recording.

These AudioSet models are not trained bolt detectors. Outputs intentionally do
not turn a generic sound tag into a bolt alarm.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from bolt_audio.config import project_root

import librosa
import numpy as np
import soundfile as sf
import torch

ROOT = project_root()
SAMPLE_RATE = 32000


def window_starts(samples: int, window: int, hop: int) -> list[int]:
    if samples <= 0 or window <= 0 or hop <= 0 or hop > window:
        raise ValueError("Require nonempty audio and 0 < hop <= window")
    last = max(0, samples - window)
    starts = list(range(0, last + 1, hop))
    if starts[-1] != last:
        starts.append(last)
    return starts


def audio_quality(audio: np.ndarray, info) -> dict:
    bits = {"PCM_16": 16, "PCM_24": 24, "PCM_32": 32, "PCM_U8": 8}.get(info.subtype)
    positive_rail = 1 - 2 ** (1 - bits) if bits else 1.0
    rail_fraction = float(np.mean((audio <= -1.0) | (audio >= positive_rail)))
    return {
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "subtype": info.subtype,
        "duration_seconds": len(audio) / info.samplerate,
        "peak_amplitude": float(np.abs(audio).max()),
        "rms_dbfs": float(20 * np.log10(max(float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))), 1e-12))),
        "full_scale_sample_fraction": rail_fraction,
        "clipping_warning": rail_fraction > 0.001,
    }


def top_tags(scores: np.ndarray, labels: list[str], top_k: int = 10) -> list[dict]:
    return [{"label": labels[int(i)], "score": float(scores[i])}
            for i in np.argsort(scores)[::-1][:top_k]]


def plot_diagnostics(path, waveform, results, sed_scores, labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = 3 if sed_scores is not None else 2
    figure, axes = plt.subplots(rows, 1, figsize=(13, rows * 2.6), constrained_layout=True)
    # Min/max envelope preserves short transients in the overview.
    stride = max(1, len(waveform) // 2500)
    usable = len(waveform) // stride * stride
    blocks = waveform[:usable].reshape(-1, stride)
    seconds = np.arange(len(blocks)) * stride / SAMPLE_RATE
    axes[0].fill_between(seconds, blocks.min(axis=1), blocks.max(axis=1), color="#176d9c", linewidth=0.3)
    axes[0].set(title="Waveform (no gain normalization)", ylabel="Amplitude", ylim=(-1.05, 1.05))
    spectrum = np.abs(librosa.stft(waveform, n_fft=1024, hop_length=320))
    db = librosa.amplitude_to_db(spectrum, ref=np.max)
    axes[1].imshow(db, origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=0,
                   extent=(0, len(waveform) / SAMPLE_RATE, 0, SAMPLE_RATE / 2000))
    axes[1].set(title="STFT magnitude (dB relative to recording maximum)", ylabel="Frequency (kHz)")
    if sed_scores is not None:
        indexes = np.argsort(sed_scores.max(axis=0))[-8:][::-1]
        axes[2].imshow(sed_scores[:, indexes].T, origin="upper", aspect="auto", cmap="viridis",
                       vmin=0, vmax=1, extent=(0, (len(sed_scores) - 1) * 0.01, 7.5, -0.5))
        axes[2].set_yticks(range(len(indexes)), [labels[i] for i in indexes])
        axes[2].set_title("Generic AudioSet scores; temporal receptive field is coarser than 10 ms")
    axes[-1].set_xlabel("Time (seconds)")
    for axis in axes:
        axis.set_xlim(0, len(waveform) / SAMPLE_RATE)
    figure.suptitle("PANNs deployment check - bolt detector has not been trained", fontsize=13)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--mode", choices=("tagging", "sed", "both"), default="both")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--hop-seconds", type=float, default=0.5)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1:
        parser.error("threads and batch-size must be positive")
    if not (1.0 <= args.window_seconds <= 10.0 and 0 < args.hop_seconds <= args.window_seconds):
        parser.error("Require 1 <= window-seconds <= 10 and 0 < hop-seconds <= window-seconds")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    from panns_inference import AudioTagging, SoundEventDetection, labels

    source = args.audio.resolve(strict=True)
    info = sf.info(source)
    original, sr = sf.read(source, dtype="float32", always_2d=True)
    if not len(original) or not np.isfinite(original).all():
        raise ValueError("Audio must contain finite, nonempty samples")
    quality = audio_quality(original, info)
    waveform = original.mean(axis=1)
    if sr != SAMPLE_RATE:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=SAMPLE_RATE)
    waveform = np.ascontiguousarray(waveform, dtype=np.float32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "audio": str(source),
        "quality": quality,
        "mode": args.mode,
        "device": args.device,
        "threads": args.threads,
        "bolt_detector_trained": False,
        "bolt_detected": None,
        "notice": "Generic AudioSet model diagnostics only. Train and validate a bolt classifier before alarming.",
        "timings_seconds": {},
    }
    arrays = {"labels": np.asarray(labels)}
    sed_scores = None

    if args.mode in ("tagging", "both"):
        print("Loading audio tagging model...", flush=True)
        mark = time.perf_counter()
        tagger = AudioTagging(checkpoint_path=str(args.checkpoint_dir / "Cnn14_mAP=0.431.pth"), device=args.device)
        result["timings_seconds"]["tagging_model_load"] = time.perf_counter() - mark
        window = round(args.window_seconds * SAMPLE_RATE)
        hop = round(args.hop_seconds * SAMPLE_RATE)
        starts = window_starts(len(waveform), window, hop)
        scores, embeddings = [], []
        mark = time.perf_counter()
        for offset in range(0, len(starts), args.batch_size):
            batch = [np.pad(waveform[s:s + window], (0, max(0, s + window - len(waveform))))
                     for s in starts[offset:offset + args.batch_size]]
            score, embedding = tagger.inference(np.stack(batch))
            scores.append(score)
            embeddings.append(embedding)
        scores = np.concatenate(scores)
        embeddings = np.concatenate(embeddings)
        result["timings_seconds"]["tagging_inference"] = time.perf_counter() - mark
        result["tagging"] = {
            "window_seconds": args.window_seconds,
            "hop_seconds": args.hop_seconds,
            "embedding_shape": list(embeddings.shape),
            "top_tags_by_window_max": top_tags(scores.max(axis=0), labels),
            "windows": [{"start_s": s / SAMPLE_RATE, "end_s": min(s + window, len(waveform)) / SAMPLE_RATE,
                         "top_tags": top_tags(score, labels, 3)} for s, score in zip(starts, scores)],
        }
        arrays.update(tagging_scores=scores, embeddings=embeddings,
                      window_start_seconds=np.asarray(starts) / SAMPLE_RATE,
                      window_end_seconds=np.minimum(np.asarray(starts) + window, len(waveform)) / SAMPLE_RATE)
        del tagger
        print(f"Tagging completed: {len(starts)} windows, embeddings {embeddings.shape}", flush=True)

    if args.mode in ("sed", "both"):
        print("Loading sound event detection model...", flush=True)
        mark = time.perf_counter()
        detector = SoundEventDetection(
            checkpoint_path=str(args.checkpoint_dir / "Cnn14_DecisionLevelMax_mAP=0.385.pth"), device=args.device)
        result["timings_seconds"]["sed_model_load"] = time.perf_counter() - mark
        chunk_size, chunk_hop = 10 * SAMPLE_RATE, 8 * SAMPLE_RATE
        feature_hop = 320
        sed_scores = np.zeros((len(waveform) // feature_hop + 1, len(labels)), dtype=np.float32)
        mark = time.perf_counter()
        for start in range(0, len(waveform), chunk_hop):
            chunk = waveform[start:start + chunk_size]
            padded = np.pad(chunk, (0, chunk_size - len(chunk)))
            prediction = detector.inference(padded[None, :])[0]
            first = start // feature_hop
            length = min(len(prediction), len(chunk) // feature_hop + 1, len(sed_scores) - first)
            sed_scores[first:first + length] = np.maximum(sed_scores[first:first + length], prediction[:length])
        result["timings_seconds"]["sed_inference"] = time.perf_counter() - mark
        result["sed"] = {"shape": list(sed_scores.shape), "output_step_seconds": 0.01,
                         "chunk_seconds": 10, "chunk_hop_seconds": 8, "overlap_merge": "maximum",
                         "notice": "Output is interpolated; 10 ms output spacing is not 10 ms event localization accuracy.",
                         "top_tags_by_frame_max": top_tags(sed_scores.max(axis=0), labels)}
        arrays.update(sed_scores=sed_scores, sed_frame_times_seconds=np.arange(len(sed_scores)) * 0.01)
        del detector
        print(f"SED completed: {sed_scores.shape}", flush=True)

    if not all(np.isfinite(value).all() for value in arrays.values() if np.issubdtype(value.dtype, np.number)):
        raise RuntimeError("Model produced non-finite values")
    array_path = args.output_dir / f"{source.stem}_features.npz"
    np.savez_compressed(array_path, **arrays)
    result["features_file"] = str(array_path.resolve())
    if not args.no_plot:
        plot_path = args.output_dir / f"{source.stem}_diagnostics.png"
        plot_diagnostics(plot_path, waveform, result, sed_scores, labels)
        result["diagnostic_plot"] = str(plot_path.resolve())
    result["timings_seconds"]["total_excluding_python_imports"] = time.perf_counter() - started
    result_path = args.output_dir / f"{source.stem}_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {result_path}", flush=True)
    print(json.dumps(result["timings_seconds"], indent=2))


if __name__ == "__main__":
    main()
