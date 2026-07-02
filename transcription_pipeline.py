"""
transcription_pipeline.py
A production-minded speech-to-text pipeline built on faster-whisper.

Capabilities
------------
1. Accepts audio in any common format (WAV, MP3, FLAC, M4A, OGG, WEBM, ...).
2. Transcribes speech to text.
3. Returns a structured result with per-segment (and optional per-word) timestamps.
4. Normalizes arbitrary input to a canonical 16 kHz mono PCM WAV via ffmpeg.
5. Handles long audio through the engine's built-in VAD segmentation, plus an
   optional fixed-window chunking mode with overlap and correct global
   timestamp offsets for very long / distributed workloads.

Design notes
------------
- The core `TranscriptionPipeline` is transport-agnostic: it is used identically
  by the CLI at the bottom of this file and by the FastAPI service in service.py.
- We decode via ffmpeg rather than trusting each container/codec's quirks. That
  gives one well-tested code path for every input format.
- The model is loaded once and reused; model init is the expensive step.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable, Optional

from faster_whisper import WhisperModel

logger = logging.getLogger("transcription")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class PipelineConfig:
    model_size: str = "base"          # tiny | base | small | medium | large-v3
    device: str = "auto"              # auto | cpu | cuda
    compute_type: str = "int8"        # int8 (CPU) | float16 (GPU) | int8_float16
    language: Optional[str] = None    # None => auto-detect
    beam_size: int = 5
    vad_filter: bool = True           # skip silence, split on speech boundaries
    word_timestamps: bool = False
    # Optional fixed-window chunking for very long files (seconds). 0 disables it;
    # faster-whisper already streams long files, so this is opt-in.
    max_chunk_seconds: int = 0
    chunk_overlap_seconds: float = 2.0
    target_sample_rate: int = 16_000
    target_channels: int = 1


# --------------------------------------------------------------------------- #
# Audio I/O and normalization
# --------------------------------------------------------------------------- #
class AudioError(RuntimeError):
    """Raised when an input cannot be validated or decoded."""


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise AudioError("ffmpeg/ffprobe not found on PATH. Install ffmpeg first.")


def probe_audio(path: Path) -> dict:
    """Return codec / sample-rate / channels / duration via ffprobe. Validates the file."""
    _require_ffmpeg()
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,sample_rate,channels:format=duration",
        "-of", "json", str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AudioError(f"Not a decodable audio file: {path.name}\n{proc.stderr.strip()}")
    info = json.loads(proc.stdout or "{}")
    if not info.get("streams"):
        raise AudioError(f"No audio stream found in {path.name}")
    stream = info["streams"][0]
    return {
        "codec": stream.get("codec_name"),
        "sample_rate": int(stream.get("sample_rate", 0) or 0),
        "channels": int(stream.get("channels", 0) or 0),
        "duration": float(info.get("format", {}).get("duration", 0.0) or 0.0),
    }


def normalize_audio(path: Path, cfg: PipelineConfig, out_dir: Path) -> Path:
    """
    Transcode any input to canonical 16 kHz mono PCM WAV.
    One code path for every format => predictable, engine-friendly input.
    """
    _require_ffmpeg()
    out_path = out_dir / f"{path.stem}.norm.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(path),
        "-ac", str(cfg.target_channels),
        "-ar", str(cfg.target_sample_rate),
        "-c:a", "pcm_s16le",
        "-vn",                       # drop any video stream (e.g. mp4/webm)
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg failed to normalize {path.name}\n{proc.stderr.strip()}")
    return out_path


# --------------------------------------------------------------------------- #
# Result data model
# --------------------------------------------------------------------------- #
@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    text: str
    language: str
    language_probability: float
    duration: float
    segments: list[Segment]

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Core pipeline
# --------------------------------------------------------------------------- #
class TranscriptionPipeline:
    def __init__(self, cfg: PipelineConfig | None = None):
        self.cfg = cfg or PipelineConfig()
        logger.info("Loading model '%s' (device=%s, compute=%s)",
                    self.cfg.model_size, self.cfg.device, self.cfg.compute_type)
        self.model = WhisperModel(
            self.cfg.model_size,
            device=self.cfg.device,
            compute_type=self.cfg.compute_type,
        )

    def transcribe(self, audio_path: str | Path) -> TranscriptionResult:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise AudioError(f"File not found: {audio_path}")

        meta = probe_audio(audio_path)
        logger.info("Input: %s | codec=%s sr=%s ch=%s dur=%.1fs",
                    audio_path.name, meta["codec"], meta["sample_rate"],
                    meta["channels"], meta["duration"])

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            wav = normalize_audio(audio_path, self.cfg, tmp_dir)

            use_chunking = (
                self.cfg.max_chunk_seconds
                and meta["duration"] > self.cfg.max_chunk_seconds
            )
            if use_chunking:
                segments, language, lang_prob = self._transcribe_long(
                    wav, tmp_dir, meta["duration"]
                )
            else:
                segments, language, lang_prob = self._transcribe_whole(wav)

        full_text = " ".join(s.text for s in segments).strip()
        return TranscriptionResult(
            text=full_text,
            language=language,
            language_probability=round(lang_prob, 4),
            duration=round(meta["duration"], 3),
            segments=segments,
        )

    # -- strategies -------------------------------------------------------- #
    def _transcribe_whole(self, wav: Path) -> tuple[list[Segment], str, float]:
        """Default path. faster-whisper streams the file in 30 s windows with VAD."""
        seg_iter, info = self.model.transcribe(
            str(wav),
            language=self.cfg.language,
            beam_size=self.cfg.beam_size,
            vad_filter=self.cfg.vad_filter,
            word_timestamps=self.cfg.word_timestamps,
        )
        segments = self._collect(seg_iter, id_offset=0, time_offset=0.0)
        return segments, info.language, info.language_probability

    def _transcribe_long(
        self, wav: Path, tmp_dir: Path, duration: float
    ) -> tuple[list[Segment], str, float]:
        """
        Explicit fixed-window chunking for very long inputs. Each window is
        transcribed independently -> bounded memory and trivial parallelism.
        Timestamps are shifted back onto the global timeline; a small overlap
        avoids dropping words at boundaries, and segments beginning inside the
        overlap tail are discarded so the next window owns them (no duplicates).
        """
        window = float(self.cfg.max_chunk_seconds)
        overlap = self.cfg.chunk_overlap_seconds

        all_segments: list[Segment] = []
        language, lang_prob = self.cfg.language or "unknown", 0.0
        start, idx = 0.0, 0

        while start < duration:
            chunk_path = tmp_dir / f"chunk_{idx:04d}.wav"
            self._slice(wav, chunk_path, start, window + overlap)

            seg_iter, info = self.model.transcribe(
                str(chunk_path),
                language=self.cfg.language,
                beam_size=self.cfg.beam_size,
                vad_filter=self.cfg.vad_filter,
                word_timestamps=self.cfg.word_timestamps,
            )
            if idx == 0:
                language, lang_prob = info.language, info.language_probability

            chunk_segments = self._collect(
                seg_iter, id_offset=len(all_segments), time_offset=start
            )
            boundary = start + window
            is_last = boundary >= duration
            for seg in chunk_segments:
                if is_last or seg.start < boundary:
                    all_segments.append(seg)

            start += window
            idx += 1

        # Renumber ids to be contiguous after overlap trimming.
        for i, seg in enumerate(all_segments):
            seg.id = i
        return all_segments, language, lang_prob

    def _slice(self, wav: Path, out: Path, start: float, length: float) -> None:
        cmd = [
            "ffmpeg", "-y", "-ss", str(start), "-t", str(length),
            "-i", str(wav), "-c", "copy", str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise AudioError(f"ffmpeg slice failed at {start}s\n{proc.stderr.strip()}")

    def _collect(
        self, seg_iter: Iterable, id_offset: int, time_offset: float
    ) -> list[Segment]:
        out: list[Segment] = []
        for i, s in enumerate(seg_iter):
            words: list[Word] = []
            if self.cfg.word_timestamps and getattr(s, "words", None):
                words = [
                    Word(start=round(w.start + time_offset, 3),
                         end=round(w.end + time_offset, 3),
                         word=w.word,
                         probability=round(w.probability, 4))
                    for w in s.words
                ]
            out.append(Segment(
                id=id_offset + i,
                start=round(s.start + time_offset, 3),
                end=round(s.end + time_offset, 3),
                text=s.text.strip(),
                words=words,
            ))
        return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audio -> text transcription with timestamps.")
    p.add_argument("audio", help="Path to an audio file (wav, mp3, flac, m4a, ogg, ...)")
    p.add_argument("--model", default="base", help="Whisper model size")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--compute-type", default="int8")
    p.add_argument("--language", default=None, help="Force language (skip auto-detect)")
    p.add_argument("--word-timestamps", action="store_true")
    p.add_argument("--max-chunk-seconds", type=int, default=0,
                   help="Enable fixed-window chunking above this duration (0=off)")
    p.add_argument("--output", "-o", default=None, help="Write JSON result to this path")
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = _build_arg_parser().parse_args()
    cfg = PipelineConfig(
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        word_timestamps=args.word_timestamps,
        max_chunk_seconds=args.max_chunk_seconds,
    )
    pipeline = TranscriptionPipeline(cfg)
    result = pipeline.transcribe(args.audio)

    payload = result.to_json()
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
        logger.info("Wrote %s", args.output)
    else:
        print(payload)


if __name__ == "__main__":
    main()
