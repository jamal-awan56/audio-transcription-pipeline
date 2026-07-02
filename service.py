"""
service.py
FastAPI wrapper exposing the transcription pipeline as a REST service.

Endpoints
---------
POST /transcribe        synchronous; best for short clips.
POST /jobs              enqueue a (long) file, returns a job id immediately (202).
GET  /jobs/{job_id}     poll job status / fetch result.

Why two paths?
--------------
An HTTP request should not block for minutes on a long transcription. Short audio
is handled inline; long audio is offloaded to a background worker so the client
gets an immediate 202 and polls for completion.

The in-memory JOBS dict + BackgroundTasks below is a stand-in for what you would
use in production: a durable store (Redis / Postgres) and a real queue + worker
pool (Celery / RQ / Cloud Tasks). That swap is the only change needed to scale
this horizontally.

Run:  uvicorn service:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from enum import Enum
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile

from transcription_pipeline import AudioError, PipelineConfig, TranscriptionPipeline

app = FastAPI(title="Transcription Pipeline", version="1.0.0")

# Model loaded once at startup and reused across all requests.
pipeline = TranscriptionPipeline(PipelineConfig(model_size="base"))


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    done = "done"
    error = "error"


JOBS: dict[str, dict] = {}  # stand-in for Redis / a real job store


def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "audio").suffix or ".bin"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with tmp:
        shutil.copyfileobj(upload.file, tmp)
    return Path(tmp.name)


def _result_to_dict(result) -> dict:
    return json.loads(result.to_json())


# NOTE: defined as a plain `def` (not `async def`) on purpose. Transcription is
# blocking CPU work, so FastAPI runs this in its threadpool and the event loop
# stays free to serve other requests. An `async def` here would block everything.
@app.post("/transcribe")
def transcribe_sync(file: UploadFile = File(...)):
    path = _save_upload(file)
    try:
        return _result_to_dict(pipeline.transcribe(path))
    except AudioError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        path.unlink(missing_ok=True)


# Plain `def` again: saving a large upload is blocking I/O, so let FastAPI
# threadpool it. The actual transcription still runs in the background task.
@app.post("/jobs", status_code=202)
def create_job(background: BackgroundTasks, file: UploadFile = File(...)):
    path = _save_upload(file)
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": JobStatus.queued, "result": None, "error": None}
    background.add_task(_run_job, job_id, path)
    return {"job_id": job_id, "status": JobStatus.queued}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job id")
    return {"job_id": job_id, **job}


def _run_job(job_id: str, path: Path) -> None:
    JOBS[job_id]["status"] = JobStatus.processing
    try:
        result = pipeline.transcribe(path)
        JOBS[job_id].update(status=JobStatus.done, result=_result_to_dict(result))
    except Exception as e:  # noqa: BLE001 - surface any failure to the client
        JOBS[job_id].update(status=JobStatus.error, error=str(e))
    finally:
        path.unlink(missing_ok=True)
