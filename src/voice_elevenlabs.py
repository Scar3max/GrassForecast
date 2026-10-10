from __future__ import annotations

import hashlib
import os
from pathlib import Path

import requests

from src.config import NUDGE_DIR


def synthesize_nudge(text: str, voice_id: str | None = None) -> str | None:
    api_key = os.getenv("ELEVENLABS_API_KEY")
    voice_id = voice_id or os.getenv("ELEVENLABS_VOICE_ID")
    if not api_key or not voice_id:
        return None

    token = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    out_path = NUDGE_DIR / f"{token}.mp3"
    if out_path.exists():
        return str(out_path)

    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }
    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
    try:
        response = requests.post(f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}", headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        out_path.write_bytes(response.content)
        return str(out_path)
    except Exception:
        return None
