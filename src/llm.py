"""Minimal client for any OpenAI-compatible chat API (Gemini, OpenRouter, vLLM, Ollama).

Settings come from environment variables, so switching models needs no code change:
  TESTER_BASE_URL  default: Gemini's OpenAI-compatible endpoint
  TESTER_API_KEY   default: $GEMINI_API_KEY
  TESTER_MODEL     default: gemini-3.5-flash (free tier of the Gemini API)
  TESTER_TPM       optional input-token budget per minute; calls wait to stay under it

  python src/llm.py models     # list the models the endpoint offers
"""

import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


def settings() -> dict:
    return {"base_url": os.environ.get("TESTER_BASE_URL", GEMINI_URL).rstrip("/"),
            "api_key": os.environ.get("TESTER_API_KEY") or os.environ.get("GEMINI_API_KEY", ""),
            "model": os.environ.get("TESTER_MODEL", "gemini-3.5-flash")}


def request(path: str, body: dict | None = None, timeout: int = 600) -> dict:
    s = settings()
    req = urllib.request.Request(s["base_url"] + path, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {s['api_key']}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def stream(body: dict, timeout: int = 900) -> dict:
    """Streamed chat call. Streaming keeps bytes flowing, so proxies don't cut long generations."""
    s = settings()
    req = urllib.request.Request(s["base_url"] + "/chat/completions",
                                 data=json.dumps({**body, "stream": True}).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {s['api_key']}"})
    text, usage = [], {}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for line in resp:
            line = line.decode(errors="replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices") or []:
                text.append((choice.get("delta") or {}).get("content") or "")
    return {"text": "".join(text), "usage": usage}


_sent: list[tuple[float, int]] = []  # (time, estimated input tokens) of recent calls


def pace(messages: list[dict]) -> None:
    """Wait until this call fits in TESTER_TPM input tokens per rolling minute (rough estimate)."""
    tpm = int(os.environ.get("TESTER_TPM") or 0)
    if not tpm:
        return
    est = sum(len(str(m.get("content", ""))) for m in messages) // 3 + 50
    while True:
        now = time.time()
        _sent[:] = [(t, n) for t, n in _sent if now - t < 62]
        if sum(n for _, n in _sent) + est <= tpm or not _sent:
            break
        time.sleep(62 - (now - _sent[0][0]) + 0.5)
    _sent.append((time.time(), est))


def chat(messages: list[dict], temperature: float = 0.2, max_tokens: int = 8192, seed: int | None = None,
         retries: int = 8) -> dict:
    """Returns {"text", "prompt_tokens", "completion_tokens"}. Waits out rate limits (429) and 5xx."""
    body = {"model": settings()["model"], "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens}
    if seed is not None:
        body["seed"] = seed
    body["stream_options"] = {"include_usage": True}
    for attempt in range(retries + 1):
        try:
            pace(messages)
            out = stream(body)
            usage = out["usage"] or {}
            text = out["text"]
            text = re.sub(r"(?s)^.*?</thought>", "", text).strip()  # Gemma returns its reasoning inline
            return {"text": text,
                    "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")}
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            unknown = [k for k in ("seed", "stream_options") if k in body and k in detail]
            if e.code == 400 and unknown:  # not every endpoint takes these
                for k in unknown:
                    del body[k]
                continue
            if (e.code == 429 or e.code >= 500) and attempt < retries:
                hint = re.search(r"retry in ([\d.]+)s", detail)
                wait = float(e.headers.get("Retry-After") or (float(hint.group(1)) + 2 if hint else min(30 * 2 ** attempt, 900)))
                print(f"  llm: HTTP {e.code}, waiting {wait:.0f}s: {' '.join(detail.split())[:200]}", flush=True)
                time.sleep(wait)
                continue
            raise RuntimeError(f"LLM request failed: HTTP {e.code}: {detail}") from e
        except (OSError, http.client.HTTPException) as e:  # dropped connections, timeouts
            if attempt < retries:
                time.sleep(min(30 * 2 ** attempt, 900))
                continue
            raise RuntimeError(f"LLM request failed: {e}") from e


if __name__ == "__main__":
    if sys.argv[1:] == ["models"]:
        for m in request("/models").get("data", []):
            print(m.get("id"))
    else:
        print(chat([{"role": "user", "content": " ".join(sys.argv[1:]) or "Say hello."}])["text"])
