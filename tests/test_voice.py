from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

from src import voice_elevenlabs


def test_synthesize_nudge_requires_credentials(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_VOICE_ID", raising=False)

    with pytest.raises(RuntimeError, match="ELEVENLABS_API_KEY"):
        voice_elevenlabs.synthesize_nudge("A short test nudge.")


def test_synthesize_nudge_saves_audio_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "test-voice")
    monkeypatch.setattr(voice_elevenlabs, "NUDGE_DIR", tmp_path)
    response = Mock()
    response.headers = {"content-type": "audio/mpeg"}
    response.content = b"test-mp3-bytes"
    response.raise_for_status.return_value = None
    request = Mock(return_value=response)
    monkeypatch.setattr(voice_elevenlabs.requests, "post", request)

    audio_path = Path(voice_elevenlabs.synthesize_nudge("A short test nudge."))

    assert audio_path.exists()
    assert audio_path.read_bytes() == b"test-mp3-bytes"
    request.assert_called_once()


def test_synthesize_nudge_surfaces_http_error_without_exposing_key(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "private-test-key")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "test-voice")
    response = Mock()
    response.status_code = 401
    response.text = '{"detail":"invalid API key"}'
    request_error = requests.HTTPError(response=response)
    failed_response = Mock()
    failed_response.raise_for_status.side_effect = request_error
    monkeypatch.setattr(voice_elevenlabs.requests, "post", Mock(return_value=failed_response))

    with pytest.raises(RuntimeError, match="HTTP 401") as exc_info:
        voice_elevenlabs.synthesize_nudge("A short test nudge.")

    assert "private-test-key" not in str(exc_info.value)


def test_synthesize_nudge_rejects_non_audio_response(tmp_path, monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "test-voice")
    monkeypatch.setattr(voice_elevenlabs, "NUDGE_DIR", tmp_path)
    response = Mock()
    response.headers = {"content-type": "application/json"}
    response.content = b'{"error":"not audio"}'
    response.raise_for_status.return_value = None
    monkeypatch.setattr(voice_elevenlabs.requests, "post", Mock(return_value=response))

    with pytest.raises(RuntimeError, match="without audio bytes"):
        voice_elevenlabs.synthesize_nudge("A short test nudge.")
