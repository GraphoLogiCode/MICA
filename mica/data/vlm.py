"""One door to a vision-language model, local-first (D3 Source B/C tooling).

Two backends, tried in this order:

  1. OLLAMA — a locally served open model on this machine's GPU (default
     qwen2.5vl:7b; override with MICA_VLM_LOCAL). No API key, no data leaves
     the machine, no cost per call. This is the preferred path.
  2. ANTHROPIC — the Claude API, only if ANTHROPIC_API_KEY is set and the local
     server is not reachable.

Callers get the model's raw text back (or None when no backend exists) plus a
`describe()` string for provenance — every artifact that carries a VLM verdict
should record which model produced it.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request

_OLLAMA = "http://localhost:11434"
_LOCAL_MODEL = os.environ.get("MICA_VLM_LOCAL", "qwen2.5vl:7b")
_API_MODEL = os.environ.get("MICA_VLM_MODEL", "claude-sonnet-5")
_FIRST_CALL_TIMEOUT = 300     # the local server loads weights into VRAM on first use


def _ollama_up() -> bool:
    try:
        with urllib.request.urlopen(f"{_OLLAMA}/api/version", timeout=2):
            return True
    except OSError:
        return False


def backend() -> str:
    """Which door is open: 'ollama', 'anthropic', or 'none'."""
    if _ollama_up():
        return "ollama"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "none"


def describe() -> str:
    chosen = backend()
    if chosen == "ollama":
        return f"ollama/{_LOCAL_MODEL} (local GPU)"
    if chosen == "anthropic":
        return f"anthropic/{_API_MODEL}"
    return "none"


def _b64(path: str) -> str:
    with open(path, "rb") as handle:
        return base64.b64encode(handle.read()).decode()


def _ask_ollama(image_paths: list[str], prompt: str, json_format: bool) -> str | None:
    body: dict = {
        "model": _LOCAL_MODEL, "stream": False,
        "options": {"temperature": 0},        # labeling wants determinism, not flair
        "messages": [{"role": "user", "content": prompt,
                      "images": [_b64(p) for p in image_paths]}],
    }
    if json_format:
        body["format"] = "json"               # the server constrains output to valid JSON
    request = urllib.request.Request(
        f"{_OLLAMA}/api/chat", data=json.dumps(body).encode(),
        headers={"content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=_FIRST_CALL_TIMEOUT) as response:
        reply = json.load(response)
    return reply.get("message", {}).get("content")


def _ask_anthropic(image_paths: list[str], prompt: str) -> str | None:
    content = [{"type": "image",
                "source": {"type": "base64",
                           "media_type": "image/png" if path.endswith(".png") else "image/jpeg",
                           "data": _b64(path)}}
               for path in image_paths]
    content.append({"type": "text", "text": prompt})
    body = json.dumps({"model": _API_MODEL, "max_tokens": 300,
                       "messages": [{"role": "user", "content": content}]}).encode()
    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body,
        headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                 "anthropic-version": "2023-06-01", "content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=90) as response:
        reply = json.load(response)
    return " ".join(block.get("text", "") for block in reply.get("content", ()))


def ask(image_paths: list[str], prompt: str, json_format: bool = False) -> str | None:
    """Show the images + prompt to whichever model is available; None when none is."""
    chosen = backend()
    try:
        if chosen == "ollama":
            return _ask_ollama(image_paths, prompt, json_format)
        if chosen == "anthropic":
            return _ask_anthropic(image_paths, prompt)
    except OSError as error:
        return f'{{"error": "{type(error).__name__}"}}' if json_format else None
    return None
