from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .env import env_bool, load_dotenv


class LLMConfigurationError(RuntimeError):
    pass


class LLMRequestError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMSettings:
    api_key: str
    base_url: str
    planner_model: str
    execution_model: str
    audit_model: str
    decision_model: str
    timeout_seconds: int
    enabled: bool
    require_llm: bool


class OpenAIResponsesClient:
    """Small Responses API client using stdlib urllib.

    This keeps the project runnable even before the OpenAI Python SDK is
    installed. It uses structured JSON outputs for every agent call.
    """

    def __init__(self, repo_root: str | Path, require_llm: bool = False, force_disable: bool = False):
        self.repo_root = Path(repo_root).resolve()
        load_dotenv(self.repo_root / ".env")
        self.settings = self._load_settings(require_llm=require_llm, force_disable=force_disable)

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    def model_for(self, agent_name: str) -> str:
        normalized = agent_name.lower()
        if "planning" in normalized:
            return self.settings.planner_model
        if "execution" in normalized:
            return self.settings.execution_model
        if "audit" in normalized:
            return self.settings.audit_model
        if "decision" in normalized:
            return self.settings.decision_model
        return self.settings.planner_model

    def structured_json(
        self,
        *,
        agent_name: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        schema_name: str,
        schema: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self.enabled:
            raise LLMConfigurationError("LLM is not enabled. Fill OPENAI_API_KEY and model names in .env.")

        model = self.model_for(agent_name)
        request_payload = {
            "model": model,
            "input": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": "Return JSON only. Input payload:\n"
                    + json.dumps(user_payload, ensure_ascii=False, indent=2),
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
        }
        response_payload = self._post_json("/responses", request_payload)
        text = self._extract_output_text(response_payload)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMRequestError(f"Model returned non-JSON output: {text[:500]}") from exc
        meta = {
            "provider": "openai",
            "api": "responses",
            "model": model,
            "schema_name": schema_name,
            "response_id": response_payload.get("id"),
            "usage": response_payload.get("usage", {}),
        }
        return parsed, meta

    def _load_settings(self, require_llm: bool, force_disable: bool) -> LLMSettings:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        default_model = os.getenv("OPENAI_MODEL", "").strip()
        planner_model = _first_nonempty(os.getenv("OPENAI_PLANNER_MODEL"), default_model)
        execution_model = _first_nonempty(os.getenv("OPENAI_EXECUTION_MODEL"), default_model, planner_model)
        audit_model = _first_nonempty(os.getenv("OPENAI_AUDIT_MODEL"), default_model, planner_model)
        decision_model = _first_nonempty(os.getenv("OPENAI_DECISION_MODEL"), default_model, planner_model)
        timeout_seconds = int(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))
        env_enabled = env_bool("SAFE_MOF_AGENT_USE_LLM", default=True)
        require_llm = require_llm or env_bool("SAFE_MOF_AGENT_REQUIRE_LLM", default=False)
        enabled = bool(api_key and planner_model and env_enabled and not force_disable)

        if require_llm and not enabled:
            raise LLMConfigurationError(
                "LLM is required but not configured. Set OPENAI_API_KEY and OPENAI_MODEL "
                "or per-agent model names in .env."
            )

        return LLMSettings(
            api_key=api_key,
            base_url=base_url,
            planner_model=planner_model,
            execution_model=execution_model or planner_model,
            audit_model=audit_model or planner_model,
            decision_model=decision_model or planner_model,
            timeout_seconds=timeout_seconds,
            enabled=enabled,
            require_llm=require_llm,
        )


def _first_nonempty(*values: str | None) -> str:
    for value in values:
        if value and value.strip():
            return value.strip()
    return ""

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = self.settings.base_url + path
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=self.settings.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code in {429, 500, 502, 503, 504} and attempt < 2:
                    time.sleep(2**attempt)
                    continue
                raise LLMRequestError(f"OpenAI API HTTP {exc.code}: {detail}") from exc
            except urllib.error.URLError as exc:
                if attempt < 2:
                    time.sleep(2**attempt)
                    continue
                raise LLMRequestError(f"OpenAI API request failed: {exc}") from exc
        raise LLMRequestError("OpenAI API request failed after retries.")

    def _extract_output_text(self, response_payload: dict[str, Any]) -> str:
        if isinstance(response_payload.get("output_text"), str):
            return response_payload["output_text"]

        texts: list[str] = []
        for item in response_payload.get("output", []):
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if isinstance(content.get("text"), str):
                        texts.append(content["text"])
            elif isinstance(item.get("content"), list):
                for content in item.get("content", []):
                    if isinstance(content.get("text"), str):
                        texts.append(content["text"])
            elif isinstance(item.get("text"), str):
                texts.append(item["text"])

        if not texts:
            raise LLMRequestError(f"No output text found in response: {response_payload}")
        return "\n".join(texts)
