from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from pipeline.cost import calculate_cost, pricing_for_provider_model
from storage.repo import StorageRepo
from viewer.api.schemas import (
    ConversationHeader,
    ConversationResponse,
    PipelineRunSummary,
)

_ARTIFACT_CATEGORY = {
    ".py": "code",
    ".urdf": "urdf",
    ".xml": "mjcf",
    ".usd": "usd",
    ".glb": "mesh",
    ".obj": "mesh",
    ".json": "blueprint",
}

_RUN_ID_OK = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-")
_TERMINAL_STATUS = {"done", "failed"}


def _is_active(status: str | None) -> bool:
    return status == "running"


def _stale_active_seconds() -> float:
    try:
        return float(os.environ.get("ARTISAN_VIEWER_STALE_SECONDS", "1800"))
    except ValueError:
        return 1800.0


def validate_run_id(run_id: str) -> bool:
    return bool(run_id) and ".." not in run_id and all(c in _RUN_ID_OK for c in run_id)


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _ts(obj: dict[str, Any]) -> float:
    try:
        return float(obj.get("ts") or 0.0)
    except (TypeError, ValueError):
        return 0.0


class ConversationBuilder:

    def __init__(self, repo: StorageRepo, run_id: str) -> None:
        self.repo = repo
        self.run_id = run_id
        if not validate_run_id(run_id):
            raise ValueError("Invalid run ID format")
        self.run_dir = repo.layout.run_dir(run_id).resolve()
        if not self.run_dir.is_relative_to(repo.root):
            raise ValueError("Run directory must be within the workspace")
        self.ws = self.run_dir / "workspace"
        if not self.ws.resolve().is_relative_to(repo.root):
            raise ValueError("Run workspace must be within the configured workspace")
        self.allow: dict[str, Path] = {}
        self.provider: str | None = None
        self.model: str | None = None
        self.pricing: dict[str, float] | None = None
        self._agent_cost: dict[str, dict[str, float]] = {}
        self._timing_costs: dict[str, list[float]] = {}
        self._timing_idx: dict[str, int] = {}
        self._timing_total: dict[str, float] = {}
        self._compaction: dict[str, dict[str, float]] = {}
        self._phase_ts: dict[str, float] = {}

    @property
    def coding_trace(self) -> Path:
        return self.ws / "trace" / "trajectory.jsonl"

    @property
    def review_trace(self) -> Path:
        return self.ws / "trace" / "review" / "trajectory.jsonl"

    @property
    def design_trace(self) -> Path:
        return self.ws / "trace" / "design.jsonl"

    @property
    def events_path(self) -> Path:
        return self.ws / "events.jsonl"

    @property
    def timing_path(self) -> Path:
        return self.ws / "trace" / "timing.jsonl"

    @property
    def state_path(self) -> Path:
        return self.run_dir / "pipeline_state.json"

    def exists(self) -> bool:
        return self.state_path.is_file() or self.ws.is_dir()

    def _token(self, image_path: str | None) -> str | None:
        if not image_path:
            return None
        abspath = Path(image_path)
        if not abspath.is_absolute():
            abspath = self.ws / abspath
        abspath = abspath.resolve()
        if not abspath.is_relative_to(self.ws.resolve()):
            return None
        token = hashlib.sha1(str(abspath).encode("utf-8")).hexdigest()[:16]
        self.allow[token] = abspath
        return token

    def _content_to_parts(self, content: Any) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        if content is None:
            return parts
        if isinstance(content, str):
            if content:
                parts.append({"type": "text", "text": content})
            return parts
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "input_image" or ptype == "image":
                    image_path = part.get("image_path") or part.get("path")
                    token = self._token(image_path)
                    if token is None:
                        continue
                    label = Path(image_path or "").stem or None
                    parts.append(
                        {
                            "type": "image",
                            "src": token,
                            "label": label,
                            "detail": part.get("detail"),
                        }
                    )
                else:
                    text = part.get("text")
                    if text:
                        parts.append({"type": "text", "text": text})
            return parts
        parts.append({"type": "text", "text": str(content)})
        return parts

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("text"):
                    chunks.append(str(part["text"]))
            return "\n".join(chunks)
        return str(content)

    def _tool_call_dict(self, raw: dict[str, Any]) -> dict[str, Any]:
        fn = raw.get("function") or {}
        return {
            "id": raw.get("id"),
            "name": fn.get("name") or raw.get("name") or "",
            "arguments": fn.get("arguments") if fn.get("arguments") is not None else "",
            "result": None,
        }

    def build(self) -> ConversationResponse:
        state = self.repo.read_json(self.state_path, default={}) or {}
        self._load_timing_costs()
        events: list[tuple[float, int, str, Any]] = []
        seq = 0

        for agent, path in (("coding", self.coding_trace), ("review", self.review_trace)):
            for obj in _iter_jsonl(path):
                kind = obj.get("type")
                if kind == "message":
                    events.append((_ts(obj), seq, f"msg:{agent}", obj.get("message")))
                    seq += 1
                elif kind and kind.startswith("compaction"):
                    self._note_compaction(agent, _ts(obj), obj)

        for obj in _iter_jsonl(self.design_trace):
            kind = obj.get("type")
            if kind == "llm_call":
                if not self.model:
                    self.provider = obj.get("provider") or self.provider
                    self.model = obj.get("model_id") or self.model
                events.append((_ts(obj), seq, "design_llm", obj))
                seq += 1
            elif kind == "design_blueprint_turn":
                if not self.model:
                    self.provider = obj.get("provider") or self.provider
                    self.model = obj.get("model_id") or self.model
                events.append((_ts(obj), seq, "design_bp", obj))
                seq += 1
            elif kind == "design_prompt":
                events.append((_ts(obj), seq, "design_prompt", obj))
                seq += 1

        board = self._load_design_board()
        if board is not None:
            events.append((board["ts"], seq, "design_board", board))
            seq += 1

        for obj in _iter_jsonl(self.events_path):
            phase = str(obj.get("phase") or "")
            if phase:
                self._phase_ts.setdefault(phase, _ts(obj))
            events.append((_ts(obj), seq, "phase", obj))
            seq += 1

        review_report = self._load_review_report()
        if review_report is not None:
            events.append((review_report["ts"], seq, "report", review_report["data"]))
            seq += 1

        if self.provider and self.model:
            self.pricing = pricing_for_provider_model(self.provider, self.model)

        for agent, agg in self._compaction.items():
            events.append((agg["last_ts"], seq, f"compaction_agg:{agent}", agg))
            seq += 1

        for gate in self._gate_events():
            events.append((gate["ts"], seq, gate["source"], gate))
            seq += 1
        artifacts = self._collect_artifacts(events, state)
        if artifacts is not None:
            events.append((artifacts["ts"], seq, "artifacts", artifacts))
            seq += 1

        events.sort(key=lambda e: (e[0], e[1]))
        earliest = events[0][0] if events else 0.0

        items: list[dict[str, Any]] = []
        design_system = self._first_design_system(events)
        if design_system:
            items.append(
                {
                    "kind": "harness_prompt",
                    "ts": earliest - 2.0,
                    "agent": "design",
                    "label": f"{_AGENT_LABEL['design']} · system prompt",
                    "parts": [{"type": "text", "text": design_system}],
                }
            )
        opening = self._opening_prompt(state, events)
        if opening is not None:
            items.append(opening)

        pending: dict[str, dict[str, Any]] = {}
        last_msg_by_agent: dict[str, dict[str, Any]] = {}

        for ts, _seq, source, payload in events:
            if source == "phase":
                items.append(self._phase_item(ts, payload))
            elif source == "report":
                items.append(self._report_item(ts, payload))
            elif source.startswith("msg:"):
                agent = source.split(":", 1)[1]
                self._handle_message(ts, agent, payload, items, pending, last_msg_by_agent)
            elif source == "design_llm":
                self._handle_design_llm(ts, payload, items)
            elif source == "design_prompt":
                self._handle_design_prompt(ts, payload, items)
            elif source == "design_bp":
                self._handle_design_blueprint(ts, payload, items)
            elif source == "design_board":
                self._handle_design_board(ts, payload, items)
            elif source.startswith("compaction_agg:"):
                items.append(self._compaction_item(ts, source.split(":", 1)[1], payload))
            elif source.startswith("gate:"):
                items.append(payload["item"])
            elif source == "artifacts":
                items.append(payload["item"])

        run_summary = self.repo.read_json(self.ws / "trace" / "run_summary.json", default={}) or {}
        seconds_by_agent = run_summary.get("active_seconds_by_agent") or {}
        active_seconds = state.get("active_seconds") or run_summary.get("active_seconds_total")
        elapsed_seconds = state.get("elapsed_seconds") or active_seconds

        last_activity = max(
            (e[0] for e in events if str(e[2]).startswith("msg:")),
            default=earliest,
        )
        is_active = _is_active(state.get("status")) and (
            last_activity <= 0.0 or (time.time() - last_activity) <= _stale_active_seconds()
        )

        header = ConversationHeader(
            run_id=self.run_id,
            status=state.get("status"),
            phase=state.get("phase"),
            termination_reason=state.get("termination_reason")
            or run_summary.get("termination_reason"),
            terminal_details=dict(
                state.get("terminal_details") or run_summary.get("terminal_details") or {}
            ),
            cost_usd=self._header_cost(state),
            elapsed_seconds=elapsed_seconds,
            active_seconds=active_seconds,
            started_at=earliest or None,
            last_activity_at=last_activity
            or None,
            last_realism_rating=state.get("last_realism_rating"),
            prompt=state.get("prompt"),
            provider=self.provider,
            model=self.model,
            is_active=is_active,
        )
        return ConversationResponse(
            header=header,
            items=items,
            cost_breakdown=self._cost_breakdown(seconds_by_agent),
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    def _handle_message(
        self,
        ts: float,
        agent: str,
        message: Any,
        items: list[dict[str, Any]],
        pending: dict[str, dict[str, Any]],
        last_msg_by_agent: dict[str, dict[str, Any]],
    ) -> None:
        if not isinstance(message, dict):
            return
        role = message.get("role")
        if role == "assistant":
            tool_calls = [
                self._tool_call_dict(tc)
                for tc in message.get("tool_calls") or []
                if isinstance(tc, dict)
            ]
            usage = message.get("usage")
            item = {
                "kind": "agent_message",
                "ts": ts,
                "agent": agent,
                "text": self._content_to_text(message.get("content")),
                "thinking": message.get("thought_summary"),
                "tool_calls": tool_calls,
                "usage": usage,
                "cost_usd": self._price_turn(agent, usage),
            }
            for tc in tool_calls:
                if tc.get("id"):
                    pending[tc["id"]] = tc
            items.append(item)
            last_msg_by_agent[agent] = item
        elif role == "tool":
            result = {"parts": self._content_to_parts(message.get("content"))}
            tcid = message.get("tool_call_id")
            target = pending.pop(tcid, None) if tcid else None
            if target is not None:
                target["result"] = result
            else:
                host = last_msg_by_agent.get(agent)
                if host is not None:
                    host["tool_calls"].append(
                        {
                            "id": tcid,
                            "name": message.get("name") or "",
                            "arguments": "",
                            "result": result,
                        }
                    )
        elif role == "user":
            parts = self._content_to_parts(message.get("content"))
            if parts:
                items.append(
                    {
                        "kind": "harness_prompt",
                        "ts": ts,
                        "agent": agent,
                        "label": _harness_label(agent),
                        "parts": parts,
                    }
                )
        elif role == "system":
            parts = self._content_to_parts(message.get("content"))
            if parts:
                items.append(
                    {
                        "kind": "harness_prompt",
                        "ts": ts,
                        "agent": agent,
                        "label": f"{_AGENT_LABEL.get(agent, agent)} · system prompt",
                        "parts": parts,
                    }
                )

    @staticmethod
    def _first_design_system(events: list[tuple[float, int, str, Any]]) -> str | None:
        for _ts, _seq, source, payload in events:
            if source in {"design_llm", "design_bp"} and isinstance(payload, dict):
                system = payload.get("system")
                if system:
                    return str(system)
        return None

    def _handle_design_llm(
        self, ts: float, obj: dict[str, Any], items: list[dict[str, Any]]
    ) -> None:
        parts = self._content_to_parts(obj.get("content"))
        if parts:
            items.append(
                {
                    "kind": "harness_prompt",
                    "ts": ts,
                    "agent": "design",
                    "label": "Design task",
                    "parts": parts,
                }
            )
        response_text = obj.get("response_text")
        if response_text:
            usage = obj.get("usage")
            items.append(
                {
                    "kind": "agent_message",
                    "ts": ts,
                    "agent": "design",
                    "text": str(response_text),
                    "thinking": None,
                    "tool_calls": [],
                    "usage": usage,
                    "cost_usd": self._price_turn("design", usage),
                }
            )

    def _handle_design_prompt(
        self, ts: float, obj: dict[str, Any], items: list[dict[str, Any]]
    ) -> None:
        parts = self._content_to_parts(obj.get("content"))
        if not parts:
            return
        label = {
            "blueprint": "Blueprint task",
            "blueprint_revision": "Blueprint revision",
            "blueprint_resume": "Blueprint resume",
            "blueprint_self_review": "Blueprint self-review",
            "intake": "Intake task",
        }.get(str(obj.get("subtask") or ""), "Design task")
        items.append(
            {
                "kind": "harness_prompt",
                "ts": ts,
                "agent": "design",
                "label": label,
                "parts": parts,
            }
        )

    def _handle_design_board(
        self, ts: float, board: dict[str, Any], items: list[dict[str, Any]]
    ) -> None:
        token = self._token(str(board["png"]))
        result_parts: list[dict[str, Any]] = []
        if token is not None:
            result_parts.append(
                {"type": "image", "src": token, "label": "design_board", "detail": None}
            )
        text = "Generated the design board (visual ground-truth for the modeling agent)."
        attempts = board.get("attempts") or []
        if attempts:
            lines = [
                f"- attempt {a.get('attempt')}: {a.get('summary')}"
                for a in attempts
                if isinstance(a, dict)
            ]
            text += f"\n\n**Board generation — {len(attempts)} attempt(s):**\n" + "\n".join(lines)
        items.append(
            {
                "kind": "agent_message",
                "ts": ts,
                "agent": "design",
                "text": text,
                "thinking": None,
                "tool_calls": [
                    {
                        "id": None,
                        "name": "generate_design_board",
                        "arguments": str(board.get("prompt") or ""),
                        "result": {"parts": result_parts},
                    }
                ],
                "usage": None,
            }
        )

    def _handle_design_blueprint(
        self, ts: float, obj: dict[str, Any], items: list[dict[str, Any]]
    ) -> None:
        assistant = obj.get("assistant") if isinstance(obj.get("assistant"), dict) else {}
        tool_calls = [
            self._tool_call_dict(tc) for tc in obj.get("tool_calls") or [] if isinstance(tc, dict)
        ]
        usage = obj.get("usage") or assistant.get("usage")
        items.append(
            {
                "kind": "agent_message",
                "ts": ts,
                "agent": "design",
                "text": self._content_to_text(assistant.get("content")),
                "thinking": assistant.get("thought_summary"),
                "tool_calls": tool_calls,
                "usage": usage,
                "cost_usd": self._price_turn("design", usage),
            }
        )

    def _phase_item(self, ts: float, obj: dict[str, Any]) -> dict[str, Any]:
        return {
            "kind": "phase_marker",
            "ts": ts,
            "phase": str(obj.get("phase") or ""),
            "iter": int(obj.get("iter") or 0),
            "status": obj.get("status"),
            "rating": obj.get("rating"),
            "next": obj.get("next"),
            "note": obj.get("note"),
            "cost_usd": obj.get("cost_usd"),
        }

    def _report_item(self, ts: float, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "kind": "review_report",
            "ts": ts,
            "report_id": data.get("report_id"),
            "phase": data.get("phase"),
            "realism_rating": data.get("realism_rating"),
            "summary": data.get("summary"),
            "decision": data.get("decision"),
            "rubric": data.get("rubric") or {},
            "issues": data.get("issues") or [],
            "issue_total": data.get("issue_total"),
        }

    def _opening_prompt(
        self, state: dict[str, Any], events: list[tuple[float, int, str, Any]]
    ) -> dict[str, Any] | None:
        prompt = state.get("prompt")
        references = state.get("references") or []
        parts: list[dict[str, Any]] = []
        if prompt:
            parts.append({"type": "text", "text": str(prompt)})
        for ref in references:
            if isinstance(ref, dict) and ref.get("media_kind", "image") == "image":
                token = self._token(ref.get("path"))
                if token is not None:
                    parts.append(
                        {
                            "type": "image",
                            "src": token,
                            "label": Path(ref.get("path", "")).stem or None,
                            "detail": None,
                        }
                    )
        legacy = state.get("image_path")
        if legacy and not references:
            token = self._token(legacy)
            if token is not None:
                parts.append(
                    {
                        "type": "image",
                        "src": token,
                        "label": Path(legacy).stem or None,
                        "detail": None,
                    }
                )
        if not parts:
            return None
        first_ts = events[0][0] if events else 0.0
        return {
            "kind": "harness_prompt",
            "ts": first_ts - 1.0,
            "agent": None,
            "label": "User request",
            "parts": parts,
        }

    def _load_timing_costs(self) -> None:
        for obj in _iter_jsonl(self.timing_path):
            if obj.get("kind") != "llm":
                continue
            agent = obj.get("agent")
            cost = obj.get("cost_usd")
            if agent is None or not isinstance(cost, (int, float)) or isinstance(cost, bool):
                continue
            key = str(agent)
            self._timing_costs.setdefault(key, []).append(float(cost))
            self._timing_total[key] = self._timing_total.get(key, 0.0) + float(cost)

    def _price_turn(self, agent: str, usage: Any) -> float | None:
        if not isinstance(usage, dict):
            return None
        bucket = self._agent_cost.setdefault(
            agent, {"turns": 0, "input": 0, "output": 0, "cached": 0, "cost": 0.0}
        )
        bucket["turns"] += 1
        bucket["input"] += int(usage.get("prompt_tokens") or 0)
        bucket["output"] += int(
            usage.get("candidates_tokens") or usage.get("completion_tokens") or 0
        )
        bucket["cached"] += int(usage.get("cached_tokens") or 0)
        queue = self._timing_costs.get(agent)
        if queue is not None:
            idx = self._timing_idx.get(agent, 0)
            if idx < len(queue):
                self._timing_idx[agent] = idx + 1
                cost = queue[idx]
                bucket["cost"] += cost
                return round(cost, 6)
        if not self.pricing:
            return None
        cost = calculate_cost(usage, self.pricing).total_cost
        bucket["cost"] += cost
        return round(cost, 6)

    def _header_cost(self, state: dict[str, Any]) -> float | None:
        snapshot = state.get("cost_usd")
        if not _is_active(state.get("status")):
            return snapshot
        if self._timing_total:
            live_total = round(sum(self._timing_total.values()), 6)
        else:
            live_total = round(sum(b.get("cost", 0.0) for b in self._agent_cost.values()), 6)
        return live_total if live_total > 0 else snapshot

    def _cost_breakdown(
        self, seconds_by_agent: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        order = {"design": 0, "coding": 1, "review": 2}
        seconds_by_agent = seconds_by_agent or {}
        rows = [
            {
                "agent": agent,
                "turns": int(b["turns"]),
                "input_tokens": int(b["input"]),
                "output_tokens": int(b["output"]),
                "cached_tokens": int(b["cached"]),
                "cost_usd": round(self._timing_total.get(agent, b["cost"]), 6),
                "active_seconds": (
                    float(seconds_by_agent[agent]) if agent in seconds_by_agent else None
                ),
            }
            for agent, b in self._agent_cost.items()
        ]
        rows.sort(key=lambda r: order.get(r["agent"], 9))
        return rows

    def _note_compaction(self, agent: str, ts: float, obj: dict[str, Any]) -> None:
        agg = self._compaction.setdefault(
            agent,
            {
                "count": 0,
                "applied": 0,
                "peak_pressure": 0.0,
                "saved": 0,
                "peak_prompt": 0,
                "last_ts": 0.0,
            },
        )
        agg["count"] += 1
        if "skip" not in str(obj.get("type") or ""):
            agg["applied"] += 1
        pr = obj.get("pressure_ratio")
        if isinstance(pr, (int, float)):
            agg["peak_pressure"] = max(agg["peak_pressure"], float(pr))
        pt = obj.get("prompt_tokens")
        if isinstance(pt, (int, float)):
            agg["peak_prompt"] = max(agg["peak_prompt"], int(pt))
        saved = obj.get("estimated_saved_next_input_tokens") or obj.get("saved_tokens")
        if isinstance(saved, (int, float)):
            agg["saved"] += int(saved)
        agg["last_ts"] = max(agg["last_ts"], ts)

    def _compaction_item(self, ts: float, agent: str, agg: dict[str, Any]) -> dict[str, Any]:
        applied = int(agg.get("applied", 0))
        return {
            "kind": "compaction",
            "ts": ts,
            "agent": agent,
            "event_type": "compaction" if applied else "compaction_skipped",
            "count": int(agg.get("count", 0)),
            "reason": f"compacted {applied}×" if applied else "monitored, no compaction needed",
            "prompt_tokens": int(agg.get("peak_prompt", 0)) or None,
            "pressure_ratio": round(float(agg.get("peak_pressure", 0.0)), 3) or None,
            "saved_tokens": int(agg.get("saved", 0)) or None,
        }

    def _gate_events(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for gate in (self._p3_gate(), self._p8_gate()):
            if gate is not None:
                out.append(gate)
        return out

    def _p3_gate(self) -> dict[str, Any] | None:
        report = self.repo.read_json(self.ws / "reports" / "P3_blueprint_review.json", default=None)
        status = summary = None
        joints: list[Any] = []
        if isinstance(report, dict):
            joints = report.get("joints") or []
            status = report.get("status")
            summary = report.get("summary")
        else:
            bp = self.repo.read_json(self.ws / "blueprint.json", default=None)
            if isinstance(bp, dict):
                joints = bp.get("joints") or []
        rows = []
        for j in joints:
            if not isinstance(j, dict):
                continue
            rows.append(
                {
                    "joint": j.get("name"),
                    "type": j.get("joint_type") or j.get("type"),
                    "axis": j.get("joint_axis") or j.get("axis"),
                    "travel": j.get("travel"),
                    "rest": j.get("rest"),
                }
            )
        if not rows:
            return None
        columns = ["joint", "type", "axis", "travel", "rest"]
        ts = self._phase_ts.get("P3_blueprint_review") or self._phase_ts.get("P1_blueprint") or 0.0
        item = {
            "kind": "gate_report",
            "ts": ts,
            "phase": "P3_blueprint_review",
            "title": "Blueprint review — joints validated",
            "status": status,
            "summary": summary,
            "columns": columns,
            "rows": rows,
        }
        return {"source": "gate:p3", "ts": ts, "item": item}

    def _p8_gate(self) -> dict[str, Any] | None:
        report = self.repo.read_json(self.ws / "reports" / "P8_export_smoke.json", default=None)
        if not isinstance(report, dict):
            return None
        ts = self._phase_ts.get("P8_export_smoke") or 0.0
        exports = report.get("exports") or {}
        rows = [
            {"format": fmt, "file": Path(str(path)).name if path else None}
            for fmt, path in exports.items()
        ]
        skipped = report.get("skipped") or []
        item = {
            "kind": "gate_report",
            "ts": ts,
            "phase": "P8_export_smoke",
            "title": "Export smoke",
            "status": "pass" if report.get("ok") else "fail",
            "summary": (f"skipped: {', '.join(map(str, skipped))}" if skipped else None),
            "columns": ["format", "file"],
            "rows": rows,
        }
        return {"source": "gate:p8", "ts": ts, "item": item}

    def _artifact_file(self, path: Path, category: str, label: str) -> dict[str, Any]:
        token = self._token(str(path))
        try:
            size = path.stat().st_size
        except OSError:
            size = None
        return {"name": path.name, "label": label, "src": token, "size": size, "category": category}

    def _collect_artifacts(
        self,
        events: list[tuple[float, int, str, Any]],
        state: dict[str, Any],
    ) -> dict[str, Any] | None:
        files: list[dict[str, Any]] = []
        for name, category in (
            ("model.py", "code"),
            ("blueprint.json", "blueprint"),
            ("qa_tests.py", "qa"),
        ):
            path = self.ws / name
            if path.is_file():
                files.append(self._artifact_file(path, category, name))
        exports_dir = self.ws / "exports"
        if exports_dir.is_dir():
            for path in sorted(exports_dir.iterdir()):
                if not path.is_file() or path.name.endswith(".materials.json"):
                    continue
                category = _ARTIFACT_CATEGORY.get(path.suffix.lower(), "other")
                files.append(self._artifact_file(path, category, path.name))
        fallback = state.get("fallback_candidate")
        fallback_status = fallback.get("status") if isinstance(fallback, dict) else None
        if fallback_status == "restored_after_failed_repair":
            snapshot_root = self.ws / "fallback_candidate"
            failed_root = self.ws / "failed_attempt"
            preferred = (
                (snapshot_root / "candidate" / "_pose_closed.glb", "fallback GLB"),
                (snapshot_root / "candidate" / "object.obj", "fallback OBJ"),
                (snapshot_root / "candidate" / "object.urdf", "fallback URDF"),
                (snapshot_root / "candidate" / "object.xml", "fallback MJCF"),
                (snapshot_root / "candidate" / "object_scene.xml", "fallback MJCF scene"),
                (snapshot_root / "candidate" / "object.usd", "fallback USD"),
                (snapshot_root / "candidate" / "manipulation.json", "fallback manipulation"),
                (snapshot_root / "candidate" / "manifest.json", "fallback manifest"),
                (snapshot_root / "candidate" / "physics_validation.json", "fallback physics"),
                (snapshot_root / "model.py", "fallback model.py"),
                (failed_root / "model.py", "failed attempt model.py"),
                (
                    failed_root / "reports" / "physics_validation.json",
                    "failed attempt physics",
                ),
                (failed_root / "failure.json", "failed attempt metadata"),
            )
            for path, label in preferred:
                if not path.is_file():
                    continue
                category = _ARTIFACT_CATEGORY.get(path.suffix.lower(), "other")
                files.append(self._artifact_file(path, category, label))
        if not files:
            return None
        ts = (max((e[0] for e in events), default=0.0)) + 1.0
        is_failed_fallback = fallback_status == "restored_after_failed_repair"
        return {
            "ts": ts,
            "item": {
                "kind": "artifacts",
                "ts": ts,
                "title": "Failed run artifacts" if is_failed_fallback else "Final artifacts",
                "status": "failed" if is_failed_fallback else None,
                "files": files,
            },
        }

    def _load_design_board(self) -> dict[str, Any] | None:
        png = self.ws / "references" / "design_board.png"
        if not png.is_file():
            return None
        meta = self.repo.read_json(self.ws / "references" / "design_board_meta.json", default={})
        meta = meta if isinstance(meta, dict) else {}
        attempts = None
        detail = meta.get("detail")
        if isinstance(detail, dict) and isinstance(detail.get("attempts"), list):
            attempts = detail["attempts"]
        return {
            "ts": png.stat().st_mtime,
            "png": png,
            "prompt": meta.get("prompt"),
            "attempts": attempts,
        }

    def _load_review_report(self) -> dict[str, Any] | None:
        reports_dir = self.ws / "reports"
        if not reports_dir.is_dir():
            return None
        candidates = sorted(reports_dir.glob("*_review.json"))
        if not candidates:
            return None
        target = max(candidates, key=lambda p: p.stat().st_mtime)
        data = self.repo.read_json(target, default=None)
        if not isinstance(data, dict):
            return None
        return {"ts": target.stat().st_mtime, "data": data}


def list_pipeline_runs(repo: StorageRepo) -> list[PipelineRunSummary]:
    runs_root = repo.layout.runs_root
    out: list[PipelineRunSummary] = []
    if not runs_root.is_dir():
        return out
    for entry in runs_root.iterdir():
        if not entry.is_dir() or not entry.resolve().is_relative_to(repo.root):
            continue
        state_path = entry / "pipeline_state.json"
        if not state_path.is_file() or not state_path.resolve().is_relative_to(repo.root):
            continue
        state = repo.read_json(state_path, default={}) or {}
        updated_at = state_path.stat().st_mtime
        out.append(
            PipelineRunSummary(
                run_id=entry.name,
                status=state.get("status"),
                phase=state.get("phase"),
                termination_reason=state.get("termination_reason"),
                terminal_details=dict(state.get("terminal_details") or {}),
                prompt=state.get("prompt"),
                cost_usd=state.get("cost_usd"),
                last_realism_rating=state.get("last_realism_rating"),
                updated_at=updated_at,
                is_active=_is_active(state.get("status"))
                and (time.time() - updated_at) <= _stale_active_seconds(),
            )
        )
    out.sort(key=lambda r: r.updated_at or 0.0, reverse=True)
    return out


_AGENT_LABEL = {"design": "DesignAgent", "coding": "CodingAgent", "review": "ReviewAgent"}


def _harness_label(agent: str) -> str:
    return {
        "coding": "Build / repair request",
        "review": "Review task",
        "design": "Design task",
    }.get(agent, "Harness prompt")
