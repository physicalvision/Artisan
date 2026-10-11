from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.cost import build_cost_tracker
from pipeline.harness_codec import MessageCodec
from pipeline.llm.factory import (
    ProviderConfig,
    create_provider_client,
)
from pipeline.providers import resolve_pipeline_model
from pipeline.timing import _null_measure

logger = logging.getLogger(__name__)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


@dataclass
class LLMClient:
    provider: str = "openai"
    model_id: str | None = None
    thinking_level: str = "high"
    trace_path: Path | None = None
    trace_label: str = ""
    cassette: Any = None
    recorder: Any = None
    openai_prompt_cache_key: str | None = None
    openai_prompt_cache_retention: str | None = None
    openai_canonical_contract_fingerprint: str | None = None

    def __post_init__(self) -> None:
        self._trace_fp = None
        self.recorder = (
            self.recorder.for_agent(self.trace_label or "design")
            if self.recorder is not None
            else None
        )
        self._llm = create_provider_client(
            ProviderConfig(
                provider=self.provider,
                model_id=resolve_pipeline_model(self.provider, self.model_id),
                thinking_level=self.thinking_level,
                openai_prompt_cache_key=self.openai_prompt_cache_key,
                openai_prompt_cache_retention=self.openai_prompt_cache_retention,
                openai_canonical_contract_fingerprint=(self.openai_canonical_contract_fingerprint),
            ),
        )
        self._codec = MessageCodec(provider=self.provider)
        self.cost = build_cost_tracker(self.provider, self._llm.model_id)

    def _open_trace(self):
        if self._trace_fp is not None or self.trace_path is None:
            return self._trace_fp
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._trace_fp = self.trace_path.open("a", encoding="utf-8")
        return self._trace_fp

    @staticmethod
    def _redact_content_for_trace(content: Any) -> Any:
        return content

    def _write_trace(
        self,
        *,
        system: str,
        content: Any,
        response_text: str,
        usage: dict[str, int] | None,
        raw_response: Any = None,
    ) -> None:
        fp = self._open_trace()
        if fp is None:
            return
        event = {
            "ts": time.time(),
            "type": "llm_call",
            "agent": self.trace_label,
            "provider": self.provider,
            "model_id": getattr(self._llm, "model_id", None),
            "system": system,
            "content": self._redact_content_for_trace(content),
            "response_text": response_text,
            "usage": usage or {},
        }
        fp.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        fp.flush()

    @property
    def cost_usd(self) -> float:
        return self.cost.all_in_total_cost_usd if self.cost else 0.0

    async def _call(self, system: str, content: Any) -> str:
        label = self.trace_label or "llm"
        if self.cassette is not None and getattr(self.cassette, "replaying", False):
            data = self.cassette.replay(label, "llm_text")
            text = str(data.get("text", ""))
            self._write_trace(
                system=system, content=content, response_text=text, usage=data.get("usage")
            )
            return text
        messages = [{"role": "user", "content": content}]
        _rec = self.recorder
        with _rec.measure() if _rec is not None else _null_measure() as _s:
            resp = await self._llm.generate_with_tools(system, messages, [])
        usage = self._codec.extract_usage(resp) or {}
        turn_cost_usd = None
        if usage and self.cost:
            turn_cost_usd = self.cost.add_turn(usage).total_cost
        if _rec is not None:
            _rec.record(
                kind="llm",
                name=label,
                sample=_s,
                tokens=usage or None,
                cost_usd=turn_cost_usd,
                cumulative_cost_usd=round(self.cost_usd, 6),
                expected_cold=False,
            )
        text = self._codec.extract_text(resp) or ""
        self._write_trace(
            system=system,
            content=content,
            response_text=text,
            usage=usage,
            raw_response=resp,
        )
        if self.cassette is not None and getattr(self.cassette, "recording", False):
            self.cassette.record(label, "llm_text", {"text": text, "usage": usage})
        return text

    async def json(self, system: str, prompt: str) -> dict[str, Any]:
        txt = await self._call(system, prompt)
        return _parse_json(txt)

    async def vision_json(
        self, system: str, prompt: str, image_paths: list[Path]
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for p in image_paths:
            content.append({"type": "input_image", "image_path": str(p), "detail": "high"})
        txt = await self._call(system, content)
        return _parse_json(txt)


def _parse_json(txt: str) -> dict[str, Any]:
    txt = txt.strip()
    m = _FENCE.search(txt)
    if m:
        txt = m.group(1).strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        pass
    start, end = txt.find("{"), txt.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(txt[start : end + 1])
        except json.JSONDecodeError:
            pass
    if start != -1:
        frag = txt[start:]
        cut = frag.rfind("},")
        if cut != -1:
            for tail in ("}]}", "}]}}", "]}", "}}", "}"):
                try:
                    return json.loads(frag[: cut + 1] + tail)
                except json.JSONDecodeError:
                    continue
    raise json.JSONDecodeError("unsalvageable JSON", txt[:200], 0)
