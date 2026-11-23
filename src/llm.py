"""Minimal client for any OpenAI-compatible chat API (Gemini, OpenRouter, vLLM, Ollama).

Settings come from environment variables, so switching models needs no code change:
  TESTER_BASE_URL  default: Gemini's OpenAI-compatible endpoint
  TESTER_API_KEY   default: $GEMINI_API_KEY
  TESTER_MODEL     default: gemini-3.5-flash (free tier of the Gemini API)
  TESTER_TPM       optional input-token budget per minute; calls wait to stay under it
  Azure OpenAI works too: point TESTER_BASE_URL at https://<resource>.../openai/v1 and the key goes in
  an api-key header.

  python src/llm.py models     # list the models the endpoint offers
"""

import http.client
import json
import os
import re
import ssl
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import truststore

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
SSL_CONTEXT = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


def settings() -> dict:
    azure = bool(os.environ.get("AZURE_OPENAI_BASE_URL")) and not os.environ.get("TESTER_BASE_URL")
    base_url = (os.environ.get("TESTER_BASE_URL") or os.environ.get("AZURE_OPENAI_BASE_URL") or GEMINI_URL).rstrip("/")
    deepseek = os.environ.get("TESTER_PROVIDER") == "deepseek" or urllib.parse.urlparse(base_url).hostname == "api.deepseek.com"
    return {"base_url": base_url,
            "api_key": os.environ.get("TESTER_API_KEY") or os.environ.get("AZURE_OPENAI_API_KEY" if azure else "DEEPSEEK_API_KEY" if deepseek else "GEMINI_API_KEY", ""),
            "model": os.environ.get("TESTER_MODEL") or (os.environ.get("AZURE_OPENAI_DEPLOYMENT") if azure else None) or "gemini-3.5-flash",
            "azure": azure,
            "deepseek": deepseek,
            "thinking": os.environ.get("TESTER_THINKING", "enabled") if deepseek else None,
            "max_tokens": int(os.environ["TESTER_MAX_TOKENS"]) if os.environ.get("TESTER_MAX_TOKENS") else None,
            "api_version": os.environ.get("AZURE_OPENAI_API_VERSION", "preview") if azure else None,
            "reasoning_effort": os.environ.get("TESTER_REASONING_EFFORT") or (os.environ.get("AZURE_OPENAI_REASONING_EFFORT") if azure else None),
            "max_completion_tokens": int(os.environ.get("TESTER_MAX_COMPLETION_TOKENS", "16384"))}


def make_request(path: str, body: dict | None = None) -> urllib.request.Request:
    s = settings()
    url = s["base_url"] + path
    if s["api_version"]:
        url += "?" + urllib.parse.urlencode({"api-version": s["api_version"]})
    auth = {"api-key": s["api_key"]} if s["azure"] else {"Authorization": f"Bearer {s['api_key']}"}
    return urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                  headers={"Content-Type": "application/json", **auth})


def headers(s: dict) -> dict:
    auth = {"api-key": s["api_key"]} if ".azure.com" in s["base_url"] else {"Authorization": f"Bearer {s['api_key']}"}
    return {"Content-Type": "application/json", **auth}


def request(path: str, body: dict | None = None, timeout: int = 600) -> dict:
    req = make_request(path, body)
    with urllib.request.urlopen(req, timeout=timeout, context=SSL_CONTEXT) as resp:
        return json.load(resp)


def stream(body: dict, timeout: int = 900) -> dict:
    """Streamed chat call. Streaming keeps bytes flowing, so proxies don't cut long generations."""
    req = make_request("/chat/completions", {**body, "stream": True})
    text, reasoning, usage, model, finish = [], [], {}, None, None
    with urllib.request.urlopen(req, timeout=timeout, context=SSL_CONTEXT) as resp:
        for line in resp:
            line = line.decode(errors="replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            usage = chunk.get("usage") or usage
            model = chunk.get("model") or model
            for choice in chunk.get("choices") or []:
                finish = choice.get("finish_reason") or finish
                text.append((choice.get("delta") or {}).get("content") or "")
                reasoning.append((choice.get("delta") or {}).get("reasoning_content") or "")
    if finish is None:
        raise OSError("Completion stream ended without a finish reason")
    return {"text": "".join(text), "reasoning_content": "".join(reasoning), "usage": usage,
            "actual_model": model, "finish_reason": finish}


_sent: list[tuple[float, int]] = []  # (time, estimated input tokens) of recent calls
_pace_lock = threading.Lock()
_cooldown_until = 0.0


def pace(messages: list[dict]) -> None:
    """Wait until this call fits in TESTER_TPM input tokens per rolling minute (rough estimate)."""
    tpm = int(os.environ.get("TESTER_TPM") or 0)
    est = sum(len(str(m.get("content", ""))) for m in messages) // 3 + 50
    while True:
        with _pace_lock:
            now = time.monotonic()
            _sent[:] = [(t, n) for t, n in _sent if now - t < 62]
            wait = max(0, _cooldown_until - now)
            if tpm and _sent and sum(n for _, n in _sent) + est > tpm:
                wait = max(wait, 62 - (now - _sent[0][0]) + 0.5)
            if wait <= 0:
                if tpm:
                    _sent.append((now, est))
                return
        time.sleep(wait)


def chat(messages: list[dict], temperature: float = 0.2, max_tokens: int = 8192, seed: int | None = None,
         retries: int = 8) -> dict:
    """Returns {"text", "prompt_tokens", "completion_tokens"}. Waits out rate limits (429) and 5xx."""
    s = settings()
    started = time.monotonic()
    body = {"model": s["model"], "messages": messages}
    if s["deepseek"]:
        if s["thinking"] not in ("enabled", "disabled"):
            raise ValueError("TESTER_THINKING must be enabled or disabled")
        body["max_tokens"] = s["max_tokens"] or max_tokens
        if not s["azure"]:
            body["thinking"] = {"type": s["thinking"]}
        elif s["thinking"] == "disabled":
            body["reasoning_effort"] = "none"
        if s["reasoning_effort"] and s["thinking"] == "enabled":
            body["reasoning_effort"] = s["reasoning_effort"]
        if s["thinking"] == "disabled":
            body["temperature"] = temperature
    elif s["reasoning_effort"]:
        body.update(reasoning_effort=s["reasoning_effort"], max_completion_tokens=s["max_completion_tokens"])
    else:
        body.update(temperature=temperature, max_tokens=s["max_tokens"] or max_tokens)
    if seed is not None and not s["deepseek"]:
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
                    "reasoning_content": out.get("reasoning_content"), "usage": usage,
                    "request_parameters": {k: v for k, v in body.items() if k != "messages"},
                    "seed_requested": seed, "seed_sent": body.get("seed"),
                    "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                    "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                    "cached_prompt_tokens": usage.get("prompt_cache_hit_tokens", (usage.get("prompt_tokens_details") or {}).get("cached_tokens")),
                    "actual_model": out.get("actual_model"), "finish_reason": out.get("finish_reason"),
                    "elapsed_seconds": round(time.monotonic() - started, 3), "retries": attempt}
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            unknown = [k for k in ("seed", "stream_options", "temperature") if k in body and k in detail]
            if e.code == 400 and unknown:  # not every endpoint takes these
                for k in unknown:
                    del body[k]
                continue
            if e.code == 400 and "max_tokens" in body and "max_completion_tokens" in detail:  # newer OpenAI models
                body["max_completion_tokens"] = max(body.pop("max_tokens"), 32768)  # includes reasoning tokens
                continue
            if (e.code == 429 or e.code >= 500) and attempt < retries:
                hint = re.search(r"retry in ([\d.]+)s", detail)
                wait = float(e.headers.get("Retry-After") or (float(hint.group(1)) + 2 if hint else min(30 * 2 ** attempt, 900)))
                print(f"  llm: HTTP {e.code}, waiting {wait:.0f}s: {' '.join(detail.split())[:200]}", flush=True)
                if e.code == 429:
                    global _cooldown_until
                    with _pace_lock:
                        _cooldown_until = max(_cooldown_until, time.monotonic() + wait)
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
