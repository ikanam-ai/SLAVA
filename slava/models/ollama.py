import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Optional

from slava.config import OLLAMA_MODEL_MAX_TOKENS, OLLAMA_MODEL_TEMPERATURE, OLLAMA_MODEL_TOP_K

try:
    from slava.config import OLLAMA_BASE_URL
except ImportError:  # backward compatibility if config.py is not replaced yet
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

try:
    from slava.config import OLLAMA_MODEL_NUM_CTX
except ImportError:
    OLLAMA_MODEL_NUM_CTX = 8192

try:
    from slava.config import OLLAMA_MODEL_TIMEOUT_SEC
except ImportError:
    OLLAMA_MODEL_TIMEOUT_SEC = 900

try:
    from slava.config import OLLAMA_MODEL_THINK
except ImportError:
    OLLAMA_MODEL_THINK = False

try:
    from slava.config import OLLAMA_SYSTEM_PROMPT
except ImportError:
    OLLAMA_SYSTEM_PROMPT = (
        "Отвечай только финальным ответом. "
        "Не выводи ход рассуждений, объяснения, анализ или промежуточные шаги. "
        "Не используй Markdown. "
        "Если это задание с выбором ответа, верни только номер/цифры ответа. "
        "Если это открытый вопрос, верни только краткий ответ."
    )


class OllamaModel:
    """Direct Ollama HTTP client for SLAVA eval.

    Why not LangChain OllamaLLM here:
    - In SLAVA eval we need explicit errors instead of silent None/NaN responses.
    - Thinking models may return reasoning separately or in the text; we request final answers only.
    - The class still exposes get_response() and invoke() so ModelHandler can call it normally.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = OLLAMA_MODEL_TEMPERATURE,
        top_k: int = OLLAMA_MODEL_TOP_K,
        num_predict: int = OLLAMA_MODEL_MAX_TOKENS,
        num_ctx: int = OLLAMA_MODEL_NUM_CTX,
        base_url: Optional[str] = None,
        timeout_sec: int = OLLAMA_MODEL_TIMEOUT_SEC,
        think: Optional[bool] = OLLAMA_MODEL_THINK,
        system_prompt: str = OLLAMA_SYSTEM_PROMPT,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.top_k = top_k
        self.num_predict = num_predict
        self.num_ctx = num_ctx
        self.base_url = (base_url or OLLAMA_BASE_URL or "http://127.0.0.1:11434").rstrip("/")
        self.timeout_sec = timeout_sec
        self.think = think
        self.system_prompt = system_prompt

    @staticmethod
    def _coerce_prompt(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            if "prompt" in value:
                return str(value["prompt"])
            if "content" in value:
                return str(value["content"])
            return str(value)
        if isinstance(value, list) and value:
            first = value[0]
            if hasattr(first, "content"):
                return str(first.content)
            if isinstance(first, dict) and "content" in first:
                return str(first["content"])
        if hasattr(value, "content"):
            return str(value.content)
        return str(value)

    @staticmethod
    def _strip_reasoning_and_markdown(text: str) -> str:
        text = str(text or "").strip()

        # Common explicit thinking blocks.
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
        text = re.sub(
            r"^\s*Thinking\.\.\..*?\.\.\.done thinking\.\s*",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()

        # Common final-answer prefixes.
        text = re.sub(r"^\s*(ответ|final answer|answer)\s*[:：\-–—]\s*", "", text, flags=re.IGNORECASE).strip()

        # Remove simple Markdown emphasis that often breaks exact match.
        text = text.replace("**", "").replace("__", "").strip()
        return text

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                raw = response.read().decode("utf-8", errors="replace")
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {raw}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama API is not available at {self.base_url}: {exc}") from exc

    def _options(self) -> dict[str, Any]:
        options = {
            "temperature": self.temperature,
            "top_k": self.top_k,
            "num_predict": self.num_predict,
        }
        if self.num_ctx:
            options["num_ctx"] = self.num_ctx
        return options

    def _chat_payload(self, prompt: str, include_think: bool) -> dict[str, Any]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": self._options(),
        }
        if include_think and self.think is not None:
            payload["think"] = bool(self.think)
        return payload

    def _generate_payload(self, prompt: str, include_think: bool) -> dict[str, Any]:
        full_prompt = prompt if not self.system_prompt else f"{self.system_prompt}\n\n{prompt}"
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": self._options(),
        }
        if include_think and self.think is not None:
            payload["think"] = bool(self.think)
        return payload

    def get_response(self, prompt: str = None) -> str:
        prompt_text = self._coerce_prompt(prompt)
        if not prompt_text.strip():
            raise ValueError("Empty prompt passed to OllamaModel")

        errors: list[str] = []

        # Try /api/chat first. If old Ollama rejects `think`, retry without it.
        for include_think in (True, False):
            try:
                data = self._post_json("/api/chat", self._chat_payload(prompt_text, include_think=include_think))
                content = data.get("message", {}).get("content", "")
                content = self._strip_reasoning_and_markdown(content)
                if content:
                    return content
                errors.append(f"/api/chat returned empty content, keys={sorted(data.keys())}")
            except Exception as exc:
                errors.append(f"/api/chat include_think={include_think}: {exc}")
                # second iteration retries without think

        # Fallback for older Ollama setups.
        for include_think in (True, False):
            try:
                data = self._post_json(
                    "/api/generate", self._generate_payload(prompt_text, include_think=include_think)
                )
                content = data.get("response", "")
                content = self._strip_reasoning_and_markdown(content)
                if content:
                    return content
                errors.append(f"/api/generate returned empty response, keys={sorted(data.keys())}")
            except Exception as exc:
                errors.append(f"/api/generate include_think={include_think}: {exc}")

        raise RuntimeError("; ".join(errors))

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> str:
        # LangChain-compatible fallback if somebody calls model.invoke(...)
        return self.get_response(self._coerce_prompt(input))
