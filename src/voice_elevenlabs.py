from __future__ import annotations

import hashlib
import os

import requests

from src.config import NUDGE_DIR


def synthesize_nudge(text: str, voice_id: str | None = None) -> str:
    api_key = os.getenv("ELEVENLABS_API_KEY")
    voice_id = voice_id or os.getenv("ELEVENLABS_VOICE_ID")
    if not api_key:
        raise RuntimeError("ElevenLabs audio was not generated: ELEVENLABS_API_KEY is not configured.")
    if not voice_id:
        raise RuntimeError("ElevenLabs audio was not generated: ELEVENLABS_VOICE_ID is not configured.")

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
        response = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        response = exc.response
        if response is not None:
            details = response.text[:500].strip()
            message = f"ElevenLabs returned HTTP {response.status_code}"
            if details:
                message += f": {details}"
        else:
            message = f"Could not reach ElevenLabs: {exc}"
        raise RuntimeError(message) from exc

    content_type = response.headers.get("content-type", "").lower()
    if not response.content or not content_type.startswith("audio/"):
        raise RuntimeError(
            "ElevenLabs returned a successful response without audio bytes "
            f"(content-type: {content_type or 'missing'})."
        )

    try:
        out_path.write_bytes(response.content)
    except OSError as exc:
        raise RuntimeError(f"Could not save the ElevenLabs audio file to {out_path}: {exc}") from exc
    return str(out_path)
