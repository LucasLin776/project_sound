from __future__ import annotations

import json
import os
import secrets
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from uuid import uuid4

import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, UploadFile

from .config import Settings
from .detector import BoltDetector


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
        return {"status": "ready", "bolt_detector_trained": request.app.state.detector.classifier is not None,
                "deployment_validated": False}

    @app.post("/detect")
    def detect(request: Request, file: UploadFile):
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
                    result = request.app.state.detector.detect(audio)
                except (ValueError, sf.LibsndfileError) as error:
                    raise HTTPException(400, str(error)) from error
            request_id = uuid4().hex
            result["request_id"] = request_id
            result["audio_name"] = Path((file.filename or "upload").replace("\\", "/")).name
            directory = cfg.project_dir / "results/api"
            directory.mkdir(parents=True, exist_ok=True)
            temporary_json = directory / f"{request_id}.json.part"
            temporary_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            temporary_json.replace(directory / f"{request_id}.json")
            return result
        finally:
            file.file.close()
            gate.release()

    return app
