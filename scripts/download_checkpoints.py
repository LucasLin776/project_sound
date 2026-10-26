"""Download and verify the two official PANNs inference checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Size and MD5 are published in https://zenodo.org/api/records/3987831.
MODELS = {
    "tagging": {
        "filename": "Cnn14_mAP=0.431.pth",
        "size": 327428481,
        "md5": "541141fa2ee191a88f24a3219fff024e",
    },
    "sed": {
        "filename": "Cnn14_DecisionLevelMax_mAP=0.385.pth",
        "size": 327428481,
        "md5": "70539c43c18b6a289b3199c503a82c5a",
    },
}


def verified(path: Path, spec: dict) -> bool:
    if not path.is_file() or path.stat().st_size != spec["size"]:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "md5").hexdigest() == spec["md5"]


def download(name: str, destination: Path) -> None:
    spec = MODELS[name]
    path = destination / spec["filename"]
    if verified(path, spec):
        print(f"Verified existing checkpoint: {path.name}", flush=True)
        return
    partial = path.with_suffix(path.suffix + ".part")
    url = f"https://zenodo.org/api/records/3987831/files/{spec['filename']}/content"
    for attempt in range(1, 4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "PANNs-local-setup/1.0"})
            count = 0
            next_report = 64 * 1024 * 1024
            print(f"Downloading {path.name}, attempt {attempt}/3", flush=True)
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
                    count += len(chunk)
                    if count >= next_report:
                        print(f"  {count / 1024**2:.0f}/{spec['size'] / 1024**2:.0f} MiB", flush=True)
                        next_report += 64 * 1024 * 1024
            if not verified(partial, spec):
                raise ValueError(f"Size or MD5 mismatch for {path.name}")
            partial.replace(path)
            print(f"Downloaded and MD5 verified: {path.name}", flush=True)
            return
        except Exception as error:
            print(f"  Download failed: {error}", flush=True)
            if attempt == 3:
                raise
            time.sleep(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS))
    parser.add_argument("--destination", type=Path, default=ROOT / "checkpoints")
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        download(name, args.destination)


if __name__ == "__main__":
    main()
