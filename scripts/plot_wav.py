"""Export waveform and STFT spectrogram PNGs from a WAV file or directory.

Example (from the project root):
    .venv/Scripts/python.exe scripts/plot_wav.py --audio data/reference/bolt_drop.wav

Audio keeps its original sample rate, amplitude, and separate channels.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save figures without requiring a desktop display.
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy import signal
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def configure_fonts() -> None:
    """Use a Chinese font when available, including for source filenames."""
    installed = set(font_manager.get_font_names())
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def compute_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    db_range: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Hz, seconds, and channel/frequency/time magnitudes in relative dB."""
    # Explicit padding keeps the requested FFT/window length for very short WAVs.
    channels = np.pad(audio.T, ((0, 0), (0, max(0, n_fft - len(audio)))))
    frequencies, times, spectrum = signal.stft(
        channels,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        boundary="zeros",
        padded=True,
        axis=-1,
    )
    magnitude = np.abs(spectrum)
    reference = float(magnitude.max())
    if reference == 0:
        # Silence should appear at the display floor, rather than as a 0 dB peak.
        decibels = np.full_like(magnitude, -db_range)
    else:
        decibels = 20 * np.log10(np.maximum(magnitude, np.finfo(magnitude.dtype).tiny))
        decibels -= 20 * np.log10(reference)
        np.clip(decibels, -db_range, 0, out=decibels)
    return frequencies, times, decibels


def plot_audio(source: Path, destination: Path, args: argparse.Namespace) -> tuple[Path, Path]:
    audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Audio must contain nonempty, finite samples")
    duration = len(audio) / sample_rate
    channel_count = audio.shape[1]
    size = (12, max(3.2, 2.6 * channel_count))
    title = f"{source.name} | {sample_rate} Hz | {duration:.3f} s"
    destination.mkdir(parents=True, exist_ok=True)
    waveform_path = destination / f"{source.stem}_waveform.png"
    spectrogram_path = destination / f"{source.stem}_spectrogram.png"

    figure, axes = plt.subplots(
        channel_count, 1, figsize=size, sharex=True, squeeze=False, layout="constrained"
    )
    try:
        limit = max(1.0, float(np.abs(audio).max())) * 1.05
        # A min/max envelope preserves brief impacts in long recordings.
        stride = max(1, math.ceil(len(audio) / 6000))
        starts = np.arange(0, len(audio), stride)
        counts = np.minimum(stride, len(audio) - starts)
        seconds = (starts + (counts - 1) / 2) / sample_rate
        for channel, axis in enumerate(axes[:, 0]):
            samples = audio[:, channel]
            if stride == 1:
                axis.plot(seconds, samples, color="#176d9c", linewidth=0.65)
                if len(samples) == 1:
                    axis.plot(seconds, samples, ".", color="#176d9c")
            else:
                low = np.minimum.reduceat(samples, starts)
                high = np.maximum.reduceat(samples, starts)
                axis.fill_between(seconds, low, high, color="#176d9c", linewidth=0.4)
                # Keep constant or silent sections visible when min == max.
                axis.plot(seconds, (low + high) / 2, color="#176d9c", linewidth=0.35)
            axis.set(ylabel=f"Ch {channel + 1}\nAmplitude", xlim=(0, duration), ylim=(-limit, limit))
            axis.grid(alpha=0.25)
        axes[-1, 0].set_xlabel("Time (s)")
        figure.suptitle(f"Waveform (original amplitude)\n{title}")
        figure.savefig(waveform_path, dpi=args.dpi)
    finally:
        plt.close(figure)

    frequencies, times, decibels = compute_spectrogram(
        audio, sample_rate, args.n_fft, args.hop_length, args.db_range
    )
    max_frequency = min(args.max_frequency or sample_rate / 2, sample_rate / 2)
    figure, axes = plt.subplots(
        channel_count, 1, figsize=size, sharex=True, squeeze=False, layout="constrained"
    )
    try:
        for channel, axis in enumerate(axes[:, 0]):
            mesh = axis.pcolormesh(
                times,
                frequencies / 1000,
                decibels[channel],
                shading="auto",
                cmap="magma",
                vmin=-args.db_range,
                vmax=0,
                rasterized=True,
            )
            axis.set(
                ylabel=f"Ch {channel + 1}\nFrequency (kHz)",
                xlim=(0, duration),
                ylim=(0, max_frequency / 1000),
            )
        axes[-1, 0].set_xlabel("Time (s)")
        figure.colorbar(mesh, ax=list(axes[:, 0]), label="Magnitude (dB relative to file peak)")
        figure.suptitle(
            f"STFT spectrogram (Hann, FFT={args.n_fft}, hop={args.hop_length})\n{title}"
        )
        figure.savefig(spectrogram_path, dpi=args.dpi)
    finally:
        plt.close(figure)
    return waveform_path, spectrogram_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True, help="WAV file or directory")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "audio_plots")
    parser.add_argument("--recursive", action="store_true", help="Include subdirectories")
    parser.add_argument("--n-fft", type=int, default=1024, help="Hann window and FFT length in samples")
    parser.add_argument("--hop-length", type=int, default=256, help="STFT frame step in samples")
    parser.add_argument("--db-range", type=float, default=80, help="Displayed range below the file peak")
    parser.add_argument("--max-frequency", type=float, help="Highest displayed Hz (up to Nyquist)")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()
    if args.n_fft < 4 or not 1 <= args.hop_length <= args.n_fft:
        parser.error("Require n-fft >= 4 and 1 <= hop-length <= n-fft")
    if not math.isfinite(args.db_range) or args.db_range <= 0 or args.dpi <= 0:
        parser.error("db-range must be finite and positive; dpi must be positive")
    if args.max_frequency is not None and (
        not math.isfinite(args.max_frequency) or args.max_frequency <= 0
    ):
        parser.error("max-frequency must be finite and positive")

    source = args.audio.resolve()
    if source.is_file():
        if source.suffix.lower() != ".wav":
            parser.error("Input file must have a .wav extension")
        sources, base = [source], source.parent
    elif source.is_dir():
        entries = source.rglob("*") if args.recursive else source.iterdir()
        sources = sorted(p for p in entries if p.is_file() and p.suffix.lower() == ".wav")
        base = source
        if not sources:
            parser.error(f"No WAV files found: {source}")
    else:
        parser.error(f"Input path does not exist: {source}")

    configure_fonts()
    output_dir = args.output_dir.resolve()
    failures = 0
    for path in sources:
        destination = output_dir / path.parent.relative_to(base)
        try:
            waveform_path, spectrogram_path = plot_audio(path, destination, args)
        except (OSError, ValueError, sf.LibsndfileError) as error:
            failures += 1
            print(f"[ERROR] {path}: {error}", file=sys.stderr, flush=True)
            continue
        print(f"[OK] {path.name}\n  {waveform_path}\n  {spectrogram_path}", flush=True)
    print(f"Completed: {len(sources) - failures} succeeded, {failures} failed.", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
