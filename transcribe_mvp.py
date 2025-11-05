#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time system audio transcription MVP.
- Captures system audio via sounddevice (use virtual loopback like VB-Cable / BlackHole / PulseAudio monitor, or WASAPI loopback on Windows).
- Splits into 5s chunks (configurable via --chunk or CHUNK_SECONDS env).
- Saves each chunk as WAV PCM16 16k mono in chunks/.
- Sends each chunk to Pollinations STT (model=openai-audio, language=uk) with retries and timeout.
- Appends transcript with timestamps to transcript.txt.
- Logs to app.log and console.
- Includes process_wav_file(filename) for local testing of STT without live capture.
"""

import argparse
import datetime as dt
import io
import json
import logging
import os
import queue
import sys
import time
from typing import Optional, Tuple

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv

# ---------------------------
# Config and logging
# ---------------------------

load_dotenv()

DEFAULT_BASE_URL = os.getenv("POLLINATIONS_BASE_URL", "https://text.pollinations.ai")
POLLINATIONS_API_TOKEN = os.getenv("POLLINATIONS_API_TOKEN", "").strip()
DEFAULT_MODEL = os.getenv("POLLINATIONS_MODEL", "openai-audio")
DEFAULT_LANGUAGE = os.getenv("POLLINATIONS_LANGUAGE", "uk")

# Files and directories
CHUNKS_DIR = "chunks"
TRANSCRIPT_FILE = "transcript.txt"
LOG_FILE = "app.log"

os.makedirs(CHUNKS_DIR, exist_ok=True)

logger = logging.getLogger("transcribe_mvp")
logger.setLevel(logging.INFO)
_formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ"
)
# Console
_console = logging.StreamHandler(sys.stdout)
_console.setFormatter(_formatter)
logger.addHandler(_console)
# File
_file = logging.FileHandler(LOG_FILE, encoding="utf-8")
_file.setFormatter(_formatter)
logger.addHandler(_file)


# ---------------------------
# Audio utils
# ---------------------------

def to_mono(audio: np.ndarray) -> np.ndarray:
    """Ensure mono by averaging channels if needed."""
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1)


def resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Simple linear resampling using numpy.interp; adequate for MVP."""
    if src_sr == dst_sr:
        return x
    duration = len(x) / float(src_sr)
    t_old = np.linspace(0, duration, num=len(x), endpoint=False)
    t_new = np.linspace(0, duration, num=int(round(duration * dst_sr)), endpoint=False)
    return np.interp(t_new, t_old, x).astype(x.dtype)


def float_to_int16(x: np.ndarray) -> np.ndarray:
    """Convert float [-1,1] to int16 PCM."""
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)


def save_wav_pcm16_mono(filename: str, audio: np.ndarray, samplerate: int):
    """Save array as WAV PCM16 mono."""
    if audio.ndim != 1:
        audio = to_mono(audio)
    sf.write(filename, audio, samplerate=samplerate, subtype='PCM_16')


def prepare_chunk(
    frames: np.ndarray,
    in_sr: int,
    target_sr: int = 16000
) -> Tuple[np.ndarray, int]:
    """
    Convert captured frames (float32, possibly stereo, in_sr) into mono PCM16 at target_sr.
    Returns (int16_audio, target_sr).
    """
    if frames.ndim > 1:
        frames = to_mono(frames)
    if in_sr != target_sr:
        frames = resample_linear(frames.astype(np.float32), in_sr, target_sr)
    pcm16 = float_to_int16(frames)
    return pcm16, target_sr


# ---------------------------
# Pollinations STT client
# ---------------------------

class PollinationsClient:
    """Client for Pollinations Speech-to-Text."""
    def __init__(self, base_url: str = DEFAULT_BASE_URL, token: str = "", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _headers(self) -> dict:
        headers = {"Accept": "*/*"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def transcribe(
        self,
        wav_bytes: bytes,
        filename: str = "audio.wav",
        model: str = DEFAULT_MODEL,
        language: str = DEFAULT_LANGUAGE,
        max_retries: int = 3
    ) -> str:
        """
        Send audio to Pollinations STT.
        Strategy:
          1) Try multipart/form-data with file + model + language.
          2) If fails with 4xx/5xx, log and raise readable error.
          3) If response is JSON with 'text', use it; else if plain text, return raw text.
        Retries with exponential backoff.
        """
        url = f"{self.base_url}/transcriptions"
        backoff = 1.0
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                files = {
                    "file": (filename, io.BytesIO(wav_bytes), "audio/wav"),
                }
                data = {
                    "model": model,
                    "language": language,
                    "lang": language,
                }
                resp = requests.post(
                    url,
                    headers=self._headers(),
                    files=files,
                    data=data,
                    timeout=self.timeout,
                )

                if 200 <= resp.status_code < 300:
                    return self._parse_text(resp)
                else:
                    logger.warning(
                        "Transcription failed (multipart) attempt %d/%d: %s %s",
                        attempt, max_retries, resp.status_code, resp.text[:500]
                    )
                    last_error = f"HTTP {resp.status_code}: {resp.text[:500]}"

            except requests.RequestException as e:
                last_error = str(e)
                logger.warning("Network error (multipart) attempt %d/%d: %s", attempt, max_retries, e)

            if attempt < max_retries:
                time.sleep(backoff)
                backoff *= 2

        raise RuntimeError(f"Transcription failed after {max_retries} attempts. Last error: {last_error}")

    @staticmethod
    def _parse_text(resp: requests.Response) -> str:
        """Parse Pollinations response as plain text or JSON with 'text' field."""
        ctype = resp.headers.get("Content-Type", "")
        text = resp.text
        if "application/json" in ctype:
            try:
                data = resp.json()
                if isinstance(data, dict) and "text" in data and isinstance(data["text"], str):
                    return data["text"]
                for key in ("transcript", "result", "data"):
                    if key in data and isinstance(data[key], str):
                        return data[key]
                return json.dumps(data, ensure_ascii=False)
            except Exception:
                return text
        return text


# ---------------------------
# Transcription pipeline
# ---------------------------

def write_transcript_entry(
    transcript_path: str,
    chunk_filename: str,
    chunk_index: int,
    start_s: float,
    end_s: float,
    text: str
):
    """Append a transcript entry with timestamp, chunk filename, and time bounds."""
    iso_ts = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    entry = (
        f"[{iso_ts}] {chunk_filename} ({start_s:.1f}s - {end_s:.1f}s)\n"
        f"Текст: {text.strip()}\n"
    )
    with open(transcript_path, "a", encoding="utf-8") as f:
        f.write(entry)


def process_wav_file(
    filename: str,
    client: Optional[PollinationsClient] = None,
    language: str = DEFAULT_LANGUAGE,
    model: str = DEFAULT_MODEL,
    timeout: float = 30.0
) -> str:
    """
    Transcribe a local WAV file (debug/testing).
    Returns transcript text.
    """
    if client is None:
        client = PollinationsClient(timeout=timeout)

    with open(filename, "rb") as f:
        wav_bytes = f.read()

    text = client.transcribe(
        wav_bytes=wav_bytes,
        filename=os.path.basename(filename),
        model=model,
        language=language,
    )
    return text


def capture_and_transcribe(
    device: Optional[str],
    samplerate: int,
    channels: int,
    chunk_seconds: float,
    target_sr: int = 16000,
    base_url: str = DEFAULT_BASE_URL,
    token: str = POLLINATIONS_API_TOKEN,
    language: str = DEFAULT_LANGUAGE,
    model: str = DEFAULT_MODEL,
    wasapi_loopback: bool = False
):
    """
    Capture audio from selected device (prefer loopback/virtual device) and transcribe in chunks.
    """
    client = PollinationsClient(base_url=base_url, token=token, timeout=45.0)

    q_frames: "queue.Queue[np.ndarray]" = queue.Queue()

    start_time = time.time()
    chunk_index = 1
    samples_per_chunk = int(samplerate * chunk_seconds)
    buffer = np.empty((0, channels), dtype=np.float32)

    extra_settings = None
    if wasapi_loopback and sys.platform.startswith("win"):
        try:
            extra_settings = sd.WasapiSettings(loopback=True)
            logger.info("Using WASAPI loopback mode.")
        except Exception as e:
            logger.warning("WASAPI loopback requested but not available: %s", e)
            extra_settings = None

    def audio_callback(indata, frames, time_info, status):
        if status:
            logger.debug("Audio status: %s", status)
        q_frames.put(indata.copy())

    try:
        devices = sd.query_devices()
        logger.info("Found %d audio devices. Use --list-devices to print them.", len(devices))
    except Exception:
        pass

    with sd.InputStream(
        device=device,
        samplerate=samplerate,
        channels=channels,
        dtype="float32",
        callback=audio_callback,
        blocksize=0,
        latency="low",
        extra_settings=extra_settings
    ):
        logger.info("Recording started (device=%s, sr=%d, ch=%d). Ctrl+C to stop.", str(device), samplerate, channels)
        while True:
            try:
                data = q_frames.get(timeout=1.0)
                if data is None:
                    continue
                buffer = np.vstack([buffer, data])

                while len(buffer) >= samples_per_chunk:
                    chunk = buffer[:samples_per_chunk]
                    buffer = buffer[samples_per_chunk:]

                    mono_int16, out_sr = prepare_chunk(chunk, in_sr=samplerate, target_sr=target_sr)

                    chunk_basename = f"chunk_{chunk_index:04d}.wav"
                    chunk_path = os.path.join(CHUNKS_DIR, chunk_basename)
                    save_wav_pcm16_mono(chunk_path, mono_int16.astype(np.int16), out_sr)

                    end_s = time.time() - start_time
                    start_s = max(0.0, end_s - chunk_seconds)

                    try:
                        with open(chunk_path, "rb") as f:
                            wav_bytes = f.read()
                        text = client.transcribe(
                            wav_bytes=wav_bytes,
                            filename=chunk_basename,
                            model=model,
                            language=language,
                            max_retries=3,
                        )
                        logger.info("Chunk %s transcribed.", chunk_basename)
                    except Exception as e:
                        text = f"[ПОМИЛКА РОЗПІЗНАВАННЯ] {e}"
                        logger.error("Transcription error for %s: %s", chunk_basename, e)

                    write_transcript_entry(TRANSCRIPT_FILE, chunk_basename, chunk_index, start_s, end_s, text)

                    chunk_index += 1

            except queue.Empty:
                continue
            except KeyboardInterrupt:
                logger.info("Interrupted by user.")
                break
            except Exception as e:
                logger.exception("Unexpected error: %s", e)
                time.sleep(0.5)


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Real-time system audio transcription MVP (Pollinations).")
    parser.add_argument("--device", type=str, default=None, help="Input device name or index. Use --list-devices to view.")
    parser.add_argument("--samplerate", type=int, default=48000, help="Capture sample rate (default 48000).")
    parser.add_argument("--channels", type=int, default=2, help="Capture channels (1=mono, 2=stereo).")
    parser.add_argument("--chunk", type=float, default=float(os.getenv("CHUNK_SECONDS", 5)), help="Chunk length in seconds (default 5).")
    parser.add_argument("--target-sr", type=int, default=16000, help="Output WAV sample rate (default 16000).")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL, help="Pollinations base URL.")
    parser.add_argument("--token", type=str, default=POLLINATIONS_API_TOKEN, help="Pollinations API token (optional).")
    parser.add_argument("--language", type=str, default=DEFAULT_LANGUAGE, help="Language code, e.g., uk.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name, e.g., openai-audio.")
    parser.add_argument("--list-devices", action="store_true", help="Print audio devices and exit.")
    parser.add_argument("--wasapi-loopback", action="store_true", help="Windows: try WASAPI loopback from output device.")
    parser.add_argument("--test-wav", type=str, default=None, help="Transcribe a local WAV file and exit.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    if args.test_wav:
        try:
            text = process_wav_file(
                filename=args.test_wav,
                language=args.language,
                model=args.model,
            )
            print("Транскрипт:\n", text)
        except Exception as e:
            logger.error("Failed to transcribe test WAV: %s", e)
        return

    try:
        capture_and_transcribe(
            device=args.device,
            samplerate=args.samplerate,
            channels=args.channels,
            chunk_seconds=args.chunk,
            target_sr=args.target_sr,
            base_url=args.base_url,
            token=args.token,
            language=args.language,
            model=args.model,
            wasapi_loopback=args.wasapi_loopback,
        )
    except KeyboardInterrupt:
        logger.info("Stopped.")
    except Exception as e:
        logger.exception("Fatal error: %s", e)


if __name__ == "__main__":
    main()
