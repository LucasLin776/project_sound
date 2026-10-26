"""Adapted from qiuqiangkong/panns_inference (MIT).

Both training and inference use the models from audioset_tagging_cnn.
Weights are loaded explicitly; importing this module never downloads data.
"""
import os
from pathlib import Path

import torch

from .config import classes_num, labels
from .models import Cnn14, Cnn14_DecisionLevelMax


def resolve_checkpoint(checkpoint_path, filename):
    root = Path(os.environ.get("BOLT_PROJECT_DIR", Path(__file__).resolve().parents[4]))
    directory = Path(os.environ.get("PANNS_CHECKPOINT_DIR", root / "checkpoints"))
    path = Path(checkpoint_path) if checkpoint_path is not None else directory / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}. Run scripts/download_checkpoints.py explicitly.")
    return path


class _Inference:
    model_class = None
    filename = ""

    def __init__(self, model=None, checkpoint_path=None, device="cpu"):
        if device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.device = device
        self.labels = labels
        self.classes_num = classes_num
        self.model = model if model is not None else self.model_class(
            sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64,
            fmin=50, fmax=14000, classes_num=classes_num,
        )
        checkpoint = torch.load(resolve_checkpoint(checkpoint_path, self.filename),
                                map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.to(device).eval()

    def _forward(self, audio):
        with torch.inference_mode():
            waveform = torch.as_tensor(audio, dtype=torch.float32, device=self.device)
            return self.model(waveform, None)


class AudioTagging(_Inference):
    model_class = Cnn14
    filename = "Cnn14_mAP=0.431.pth"

    def inference(self, audio):
        output = self._forward(audio)
        return output["clipwise_output"].cpu().numpy(), output["embedding"].cpu().numpy()


class SoundEventDetection(_Inference):
    model_class = Cnn14_DecisionLevelMax
    filename = "Cnn14_DecisionLevelMax_mAP=0.385.pth"

    def __init__(self, model=None, checkpoint_path=None, device="cpu", interpolate_mode="nearest"):
        if interpolate_mode != "nearest":
            raise ValueError("The upstream model supports nearest interpolation in this project")
        super().__init__(model=model, checkpoint_path=checkpoint_path, device=device)

    def inference(self, audio):
        return self._forward(audio)["framewise_output"].cpu().numpy()
