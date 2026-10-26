"""Compatibility imports for scripts originally using panns_inference."""
from bolt_audio.vendor.panns.config import labels
from bolt_audio.vendor.panns.inference import AudioTagging, SoundEventDetection

__all__ = ["AudioTagging", "SoundEventDetection", "labels"]
