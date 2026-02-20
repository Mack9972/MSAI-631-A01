from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from typing import Any, Optional

from app.core.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
)


def azure_openai_available() -> bool:
    return bool(
        AZURE_OPENAI_ENDPOINT
        and AZURE_OPENAI_API_KEY
        and AZURE_OPENAI_DEPLOYMENT
    )


def chat_completion(messages: list[dict[str, str]], temperature: float = 0.2, max_tokens: int = 400) -> Optional[str]:
    if not azure_openai_available():
        return None

    url = (
        f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/"
        f"{AZURE_OPENAI_DEPLOYMENT}/chat/completions"
        f"?api-version={AZURE_OPENAI_API_VERSION}"
    )
    payload: dict[str, Any] = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "api-key": AZURE_OPENAI_API_KEY,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
        response = json.loads(body)
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        print(f"[azure-openai] HTTP {exc.code}: {error_body}", file=sys.stderr)
        return None
    except urllib.error.URLError as exc:
        print(f"[azure-openai] Connection error: {exc}", file=sys.stderr)
        return None
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"[azure-openai] Invalid response: {exc}", file=sys.stderr)
        return None

    choices = response.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not content:
        return None
    return content.strip()
