"""Start a temporary local server, test real uploads, and stop the owned process."""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional server config containing your trained classifier")
    args = parser.parse_args()
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
        command = [sys.executable, "-m", "bolt_audio"]
        if args.config:
            command.extend(["--config", str(args.config.resolve())])
        command.extend(["serve", "--host", "127.0.0.1", "--port", str(port)])
        process = subprocess.Popen(
            command,
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
                logs = []
                for index in range(2):
                    with (root / "data/reference/bolt_drop.wav").open("rb") as source:
                        response = client.post("/detect", files={"file": ("bolt_drop.wav", source, "audio/wav")})
                    request_id = response.headers["X-Request-ID"]
                    detail = json.loads((root / f"logs/api/{request_id}.json").read_text(encoding="utf-8"))
                    if health.json()["bolt_detector_trained"]:
                        response.raise_for_status()
                        result = response.json()
                        if set(result) != {"bolt_detected", "events", "score", "quality", "processing_seconds"}:
                            raise AssertionError("Unexpected public detection contract")
                        if type(result["bolt_detected"]) is not bool or detail["result"] != result:
                            raise AssertionError("Result/log mismatch")
                        print(f"Upload {index + 1}/2 completed: {result['processing_seconds']:.2f}s", flush=True)
                    else:
                        if response.status_code != 503 or detail["error_type"] != "UntrainedDetectorError":
                            raise AssertionError("Missing target classifier must fail explicitly")
                        print(f"Upload {index + 1}/2 correctly refused: no trained classifier", flush=True)
                    responses.append({"request_id": request_id, "status_code": response.status_code,
                                      "body": response.json()})
                    logs.append(detail)
                if health.json()["bolt_detector_trained"] and logs[0]["model_load_seconds"] != logs[1]["model_load_seconds"]:
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
