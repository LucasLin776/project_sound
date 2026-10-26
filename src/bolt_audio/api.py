from __future__ import annotations

import os
import secrets
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from uuid import uuid4

import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, Response, UploadFile

from .config import Settings
from .detector import BoltDetector, UntrainedDetectorError
from .reporting import write_json


def create_app(settings: Settings | None = None, detector_factory=BoltDetector) -> FastAPI:
    cfg = settings or Settings.load(Path(os.environ["BOLT_CONFIG"]) if os.environ.get("BOLT_CONFIG") else None)
    gate = Lock()
    token = os.environ.get("BOLT_API_TOKEN")

    @asynccontextmanager
    async def lifespan(app):
        # Load once; synchronous endpoints execute in FastAPI's thread pool.
        app.state.detector = detector_factory(cfg)
        yield
        app.state.detector = None

    app = FastAPI(title="Wind turbine bolt audio prototype", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request):
        trained = request.app.state.detector.classifier is not None
        return {"status": "ready" if trained else "model_missing", "bolt_detector_trained": trained,
                "deployment_validated": False}

    @app.post("/detect")
    def detect(request: Request, response: Response, file: UploadFile):
        if token and not secrets.compare_digest(request.headers.get("authorization", ""), f"Bearer {token}"):
            raise HTTPException(401, "Invalid API token")
        if not gate.acquire(blocking=False):
            raise HTTPException(503, "Detector busy; retry later", headers={"Retry-After": "5"})
        try:
            with tempfile.TemporaryDirectory(prefix="bolt-audio-") as temporary:
                audio = Path(temporary) / "upload.wav"
                size = 0
                with audio.open("wb") as stream:
                    while chunk := file.file.read(1024 * 1024):
                        size += len(chunk)
                        if size > cfg.max_upload_bytes:
                            raise HTTPException(413, "Uploaded file exceeds size limit")
                        stream.write(chunk)
                if not size:
                    raise HTTPException(400, "Empty audio file")
                try:
                    request_id = uuid4().hex
                    result = request.app.state.detector.detect(
                        audio, log_path=cfg.project_dir / "logs/api" / f"{request_id}.json",
                        request_id=request_id,
                        audio_name=Path((file.filename or "upload").replace("\\", "/")).name,
                    )
                except UntrainedDetectorError as error:
                    raise HTTPException(503, str(error), headers={"X-Request-ID": request_id}) from error
                except (ValueError, sf.LibsndfileError) as error:
                    raise HTTPException(400, str(error), headers={"X-Request-ID": request_id}) from error
            write_json(cfg.project_dir / "results/api" / f"{request_id}.json", result)
            response.headers["X-Request-ID"] = request_id
            return result
        finally:
            file.file.close()
            gate.release()

    return app
