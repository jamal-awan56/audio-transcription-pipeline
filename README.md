Audio Transcription Pipeline
A production-minded speech-to-text service that converts audio into timestamped text for downstream use. Built on faster-whisper, with a transport-agnostic core, a CLI, and an async REST service.

Focus: engineering decisions, not model training. This README explains why the system is built the way it is.

1. What it does
Accepts audio in any common format (WAV, MP3, FLAC, M4A, OGG, WEBM, …).
Validates and normalizes the input, then transcribes speech to text.
Returns structured JSON with per-segment timestamps (and optional per-word timestamps).
Handles long audio via voice-activity detection, optional overlap chunking, and an async job model.
Exposes both a CLI and a REST API.
2. Architecture
                 ┌──────────────┐
   audio  ─────► │  API / CLI   │   (thin transport layer)
                 └──────┬───────┘
                        │  (long files → async job)
                        ▼
                 ┌──────────────┐        ┌─────────────┐
                 │ Job Queue    │ ─────► │  Worker(s)  │
                 │ (Redis/RQ)   │        │  pool       │
                 └──────────────┘        └──────┬──────┘
                                                │
                        ┌───────────────────────┼───────────────────────┐
                        ▼                       ▼                       ▼
                ┌──────────────┐        ┌──────────────┐        ┌──────────────┐
                │ ffprobe      │        │  ffmpeg      │        │ faster-whisper│
                │ (validate)   │        │ (normalize)  │        │ (transcribe)  │
                └──────────────┘        └──────────────┘        └──────────────┘
                        │                                              │
                        ▼                                              ▼
                 Object storage (S3)  for audio        DB + object storage for transcripts
The core (TranscriptionPipeline) is transport-agnostic — the exact same object powers the CLI and the API. This is the single most important structural decision: STT logic stays testable and reusable regardless of how it's invoked.

3. Key design decisions
Engine: faster-whisper (not vanilla Whisper)
OpenAI Whisper reimplemented on CTranslate2 — roughly 4× faster, lower memory, int8 quantization for CPU, and native segment + word timestamps with built-in VAD. Same accuracy as Whisper, far better runtime characteristics. Open-source (MIT), so it satisfies the "open-source library" requirement with no per-call cost or vendor lock-in.

Alternatives considered: cloud APIs (Deepgram, AWS Transcribe, AssemblyAI) — excellent but paid and less controllable; whisper.cpp — great for edge/CPU but fewer batteries included. faster-whisper is the best balance of quality, speed, and control for a self-hosted service.

Decode everything through ffmpeg
Rather than special-casing each format, every input is normalized to one canonical format: 16 kHz, mono, PCM WAV — exactly what Whisper expects internally. One well-tested code path instead of N brittle ones; predictable quality; new formats supported for free as long as ffmpeg can decode them. ffprobe validates the file up front so bad inputs fail fast with a clear error.

Async job model for long / concurrent work
An HTTP request must never block for minutes. Long files are enqueued and processed by a worker pool; the client gets an immediate job id and polls (or receives a webhook). This decouples ingestion from compute and is what makes the system scale.

Structured, typed output
Results are dataclasses serialized to a stable JSON schema (text, language, duration, segments[]). Downstream consumers get a contract, not free-form text.

4. Project structure
.
├── transcription_pipeline.py   # Core library + CLI (validation, normalize, transcribe, chunking)
├── service.py                  # FastAPI service: sync endpoint + async job endpoints
├── requirements.txt            # Python deps (ffmpeg is a system dependency)
└── README.md
5. Setup
# 1. System dependency
#    macOS:  brew install ffmpeg
#    Ubuntu: sudo apt-get install -y ffmpeg

# 2. Python deps
pip install -r requirements.txt
6. Usage
CLI
# Basic
python transcription_pipeline.py sample.mp3

# Word-level timestamps, write JSON to a file
python transcription_pipeline.py sample.mp3 --word-timestamps -o out.json

# Force language, larger model
python transcription_pipeline.py talk.m4a --language en --model small

# Very long file → explicit overlap chunking
python transcription_pipeline.py lecture.wav --max-chunk-seconds 300
REST API
uvicorn service:app --host 0.0.0.0 --port 8000
Method	Endpoint	Purpose
POST	/transcribe	Synchronous — best for short clips
POST	/jobs	Enqueue a long file, returns a job id (202)
GET	/jobs/{job_id}	Poll status / fetch result
# Sync
curl -F "file=@sample.mp3" http://localhost:8000/transcribe

# Async
curl -F "file=@long.wav" http://localhost:8000/jobs        # -> {"job_id": "...", ...}
curl http://localhost:8000/jobs/<job_id>
Sample output
{
  "text": "Hello and welcome.",
  "language": "en",
  "language_probability": 0.98,
  "duration": 4.21,
  "segments": [
    { "id": 0, "start": 0.0, "end": 4.21, "text": "Hello and welcome.", "words": [] }
  ]
}
Mock data
No audio on hand?

ffmpeg -f lavfi -i sine=frequency=440:duration=5 sample.wav   # tone, exercises the format/validation path
# or generate speech with any TTS for a realistic transcript
7. Handling different audio formats
Normalization-first (see §3). ffprobe inspects codec/sample-rate/channels/ duration and rejects non-audio or corrupt files early; ffmpeg transcodes to 16 kHz mono PCM WAV. The model therefore always sees identical input, and format variety never reaches the transcription logic.

8. Handling long audio files
Layered strategy:

VAD skips silence and splits on speech boundaries; the engine streams the file in 30 s windows instead of loading it whole.
Overlap chunking (--max-chunk-seconds) for very long/distributed jobs: fixed windows with a small overlap, each transcribed independently, timestamps shifted back to the global timeline, overlap duplicates dropped. Bounds memory and enables parallelism.
Async jobs at the service layer so long transcriptions never block a request.
9. System design notes (scaling to production)
Concurrent uploads. Stateless API servers behind a load balancer; uploads land in object storage (S3) via presigned URLs so the API never becomes a bottleneck. Work goes onto a queue (Redis + RQ/Celery, or SQS); a horizontally scalable worker pool consumes it. Concurrency per worker is bounded by GPU/CPU memory. Backpressure via queue-depth limits and 429 responses; idempotency by content hash so duplicate uploads don't reprocess.

Storage. Audio in object storage (durable, cheap, lifecycle policies to cold-storage/expire). Transcript metadata + segments in Postgres (JSONB for the segment array), with large transcripts optionally in object storage referenced by a DB pointer. Full-text search via OpenSearch/Elasticsearch. Encryption at rest and access control because audio/transcripts often contain PII.

Retry & recovery. Idempotent jobs keyed by content hash. Automatic retries with exponential backoff + jitter, capped attempts. Transient failures (OOM, worker crash) retry; permanent ones (corrupt/unsupported file) fail fast with a reason. Queue visibility-timeout re-delivers jobs from dead workers. Exhausted jobs go to a dead-letter queue with alerting. Per-chunk checkpointing so a long file resumes instead of restarting.

API surface. Versioned REST (/v1/), async job + webhook callbacks to avoid polling, presigned uploads for large files, transcript export formats (JSON, SRT, VTT, TXT), API-key/OAuth2 auth, per-client rate limits, auto-generated OpenAPI docs, consistent error schema, idempotency keys.

10. Future improvements
Speaker diarization (who spoke when).
Streaming/real-time transcription over WebSocket.
Confidence-based filtering and profanity/PII redaction.
Batched GPU inference for higher throughput.
Observability: structured logs, Prometheus metrics, tracing.
CI with a mocked test suite (no real audio required).
11. Tech stack
Python · faster-whisper (CTranslate2) · ffmpeg · FastAPI · Uvicorn
