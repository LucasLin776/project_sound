"""Start a temporary local server, test real uploads, and stop the owned process."""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "results/server_check"
    output.mkdir(parents=True, exist_ok=True)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    headers = {}
    if os.environ.get("BOLT_API_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['BOLT_API_TOKEN']}"
    with (output / "server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "bolt_audio", "serve", "--host", "127.0.0.1", "--port", str(port)],
            cwd=root, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            # Keep loopback requests local even when the workstation has proxy variables.
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=60, headers=headers,
                              trust_env=False) as client:
                deadline = time.monotonic() + 55
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"Server exited; inspect {output / 'server.log'}")
                    try:
                        health = client.get("/health")
                        health.raise_for_status()
                        break
                    except httpx.ConnectError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Local server failed to become ready within 55 seconds")
                        time.sleep(0.25)
                print(f"Server ready at 127.0.0.1:{port}", flush=True)
                responses = []
                for index in range(2):
                    with (root / "data/reference/bolt_drop.wav").open("rb") as source:
                        response = client.post("/detect", files={"file": ("bolt_drop.wav", source, "audio/wav")})
                    response.raise_for_status()
                    result = response.json()
                    if result["bolt_detected"] is not None or result["bolt_detector_trained"]:
                        raise AssertionError("Default reference checks must not claim a trained bolt detector")
                    if result["feature_shape"] != [58, 2060]:
                        raise AssertionError(f"Unexpected reference feature shape: {result['feature_shape']}")
                    responses.append(result)
                    print(f"Upload {index + 1}/2 completed: {result['processing_seconds']:.2f}s", flush=True)
                if responses[0]["model_load_seconds"] != responses[1]["model_load_seconds"]:
                    raise AssertionError("Server model was reloaded between requests")
                if responses[0]["request_id"] == responses[1]["request_id"]:
                    raise AssertionError("Request IDs must be unique")
                (output / "check.json").write_text(json.dumps({"health": health.json(), "uploads": responses},
                                                             ensure_ascii=False, indent=2), encoding="utf-8")
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
    print("Local HTTP check passed; temporary server stopped.", flush=True)


if __name__ == "__main__":
    main()
