"""Minimal Gemini client for Vertex AI (Task 6.2).

Calls the ``generateContent`` REST endpoint with JSON-schema output. It
authenticates with the local gcloud login (``gcloud auth print-access-token``),
so no API key is stored. The project comes from ``GOOGLE_CLOUD_PROJECT`` or
the active gcloud configuration.

Settings follow Google's Gemini 3 guidance: temperature stays at the default
1.0 (lower values can cause looping) and reasoning depth is set with
``thinking_level``. A fixed seed makes repeated calls more similar but does
not make them deterministic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import shutil
import subprocess
import threading
import time
from typing import Any, Mapping

import requests

TOKEN_LIFETIME_SECONDS = 30 * 60
RETRY_STATUS = {429, 500, 502, 503, 504}


class GenerationError(RuntimeError):
    """The model call failed after retries."""


@dataclass(frozen=True)
class GeminiConfig:
    model: str = "gemini-3.8-flash"
    project: str | None = None
    location: str = "global"
    temperature: float = 1.0
    seed: int = 0
    max_output_tokens: int = 16384
    thinking_level: str = "MEDIUM"
    timeout_seconds: int = 180
    max_attempts: int = 5

    def settings(self) -> dict[str, Any]:
        """The generation settings recorded with every output (no project ID)."""
        recorded = asdict(self)
        recorded.pop("project")
        return recorded


@dataclass(frozen=True)
class Generation:
    text: str
    finish_reason: str | None
    model_version: str | None
    usage: dict[str, Any]
    latency_seconds: float
    attempts: int


def _gcloud(*args: str) -> str:
    executable = shutil.which("gcloud")
    if executable is None:
        raise GenerationError("gcloud is not installed or not on PATH")
    result = subprocess.run([executable, *args], capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise GenerationError(f"gcloud {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def default_project() -> str:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or _gcloud("config", "get-value", "project")
    if not project:
        raise GenerationError("no Google Cloud project: set GOOGLE_CLOUD_PROJECT or run gcloud config set project")
    return project


class GeminiClient:
    def __init__(self, config: GeminiConfig = GeminiConfig(), session: requests.Session | None = None):
        self.config = config
        self.project = config.project or default_project()
        self._session = session or requests.Session()
        self._token: str | None = None
        self._token_time = 0.0
        self._lock = threading.Lock()

    @property
    def endpoint(self) -> str:
        host = "aiplatform.googleapis.com" if self.config.location == "global" else f"{self.config.location}-aiplatform.googleapis.com"
        return (f"https://{host}/v1/projects/{self.project}/locations/{self.config.location}"
                f"/publishers/google/models/{self.config.model}:generateContent")

    def _access_token(self, refresh: bool = False) -> str:
        with self._lock:
            if refresh or self._token is None or time.time() - self._token_time > TOKEN_LIFETIME_SECONDS:
                self._token = _gcloud("auth", "print-access-token")
                self._token_time = time.time()
            return self._token

    def request_body(self, system: str, user: str, schema: Mapping[str, Any]) -> dict[str, Any]:
        c = self.config
        return {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": c.temperature,
                "seed": c.seed,
                "maxOutputTokens": c.max_output_tokens,
                "responseMimeType": "application/json",
                "responseJsonSchema": schema,
                "thinkingConfig": {"thinkingLevel": c.thinking_level},
            },
        }

    def generate(self, system: str, user: str, schema: Mapping[str, Any]) -> Generation:
        body = self.request_body(system, user, schema)
        started = time.time()
        last_error = ""
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                response = self._session.post(
                    self.endpoint, json=body, timeout=self.config.timeout_seconds,
                    headers={"Authorization": f"Bearer {self._access_token(refresh=attempt > 1 and '401' in last_error)}"},
                )
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == 200:
                    return _parse_response(response.json(), time.time() - started, attempt)
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                if response.status_code not in RETRY_STATUS and response.status_code != 401:
                    break
            if attempt < self.config.max_attempts:
                time.sleep(min(60, 2 ** attempt))
        raise GenerationError(last_error)


def _parse_response(data: Mapping[str, Any], latency: float, attempts: int) -> Generation:
    candidates = data.get("candidates") or []
    if not candidates:
        feedback = json.dumps(data.get("promptFeedback", {}))
        raise GenerationError(f"no candidates returned (prompt feedback: {feedback})")
    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts if not part.get("thought"))
    return Generation(
        text=text,
        finish_reason=candidate.get("finishReason"),
        model_version=data.get("modelVersion"),
        usage={key: value for key, value in (data.get("usageMetadata") or {}).items() if isinstance(value, int)},
        latency_seconds=round(latency, 2),
        attempts=attempts,
    )
