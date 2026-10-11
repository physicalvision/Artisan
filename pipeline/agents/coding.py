from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from pipeline.agents._references import reference_caption_block, reference_captions
from pipeline.agents.prompts.coding import load_coding_prompt
from pipeline.context_policy import (
    boundary_compact,
    elide_stale_motion_selfcheck_nudges,
    reset_tier_a_dedup,
)
from pipeline.contracts import Issue
from pipeline.harness_sdk import HarnessResult, SdkCodingHarness

logger = logging.getLogger(__name__)


def _strip_dangling_tool_calls(conv: list[dict]) -> list[dict]:
    answered = {m.get("tool_call_id") for m in conv if m.get("role") == "tool"}
    out = list(conv)
    while out and out[-1].get("role") == "assistant":
        ids = [tc.get("id") for tc in (out[-1].get("tool_calls") or [])]
        if ids and any(cid not in answered for cid in ids):
            out.pop()
        else:
            break
    return out


def _format_issue(i: Issue) -> str:
    where = ""
    if i.code_ref:
        loc = i.code_ref.file
        if i.code_ref.symbol:
            loc += f":{i.code_ref.symbol}"
        if i.code_ref.line:
            loc += f":{i.code_ref.line}"
        where = f" @ {loc}"
    lines = [
        f"[{i.issue_id}] {i.severity.value} · {i.phase} · target={i.target}{where}",
        f"    problem:  {i.problem}",
        f"    required: {i.required_change}",
    ]
    if i.evidence:
        lines.append(f"    evidence: {'; '.join(i.evidence)}")
    return "\n".join(lines)


class CodingAgent:
    def __init__(
        self,
        model_path: Path,
        corpus_root: Path,
        session_path: Path,
        *,
        sdk_package: str = "sdk",
        max_cost_usd: float | None = 10.0,
        trace_dir: Path | None = None,
        cassette: object | None = None,
        backend: object | None = None,
        render_root: Path | None = None,
        recorder: object | None = None,
    ) -> None:
        from pipeline.providers import pipeline_model, pipeline_provider, pipeline_thinking

        self.session_path = session_path
        self.canonical_session_path = session_path.with_name("coding.canonical.json")
        self.harness = SdkCodingHarness(
            model_path,
            corpus_root=corpus_root,
            sdk_package=sdk_package,
            provider=pipeline_provider("coding"),
            model_id=pipeline_model("coding"),
            thinking_level=pipeline_thinking("coding"),
            max_cost_usd=max_cost_usd,
            trace_dir=trace_dir,
            cassette=cassette,
            backend=backend,
            render_root=render_root,
            recorder=recorder,
        )

    def _load_conv(self) -> list[dict]:
        if not self.session_path.exists():
            return []
        raw_text = self.session_path.read_text(encoding="utf-8")
        parsed = json.loads(raw_text)
        conv = _strip_dangling_tool_calls(parsed)
        conv, elided_motion_selfchecks = elide_stale_motion_selfcheck_nudges(conv)
        checkpoint_enabled = self._canonical_checkpoint_enabled()
        neutral_changed = conv != parsed
        if neutral_changed:
            self.canonical_session_path.unlink(missing_ok=True)
            self._reset_provider_history()
            reset_tier_a_dedup(self.harness)
            self.harness._expect_cold_next_call = True
        if not checkpoint_enabled:
            self.canonical_session_path.unlink(missing_ok=True)
        elif neutral_changed:
            if elided_motion_selfchecks:
                logger.info(
                    "invalidated canonical checkpoint after eliding %d stale motion nudges",
                    elided_motion_selfchecks,
                )
        elif self.canonical_session_path.exists():
            try:
                checkpoint = json.loads(self.canonical_session_path.read_text(encoding="utf-8"))
                digest = hashlib.sha1(raw_text.encode("utf-8")).hexdigest()
                imported = self.harness.llm.import_canonical_state(
                    checkpoint,
                    conversation_digest=digest,
                    conversation_message_count=len(conv),
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                imported = False
            if imported:
                reset_tier_a_dedup(self.harness)
                if bool(getattr(self.harness.llm, "_last_import_replaced_state", False)):
                    self.harness._expect_cold_next_call = True
                return conv
            self.canonical_session_path.unlink(missing_ok=True)
            self._reset_provider_history()
            self.harness._expect_cold_next_call = True

        return boundary_compact(self.harness, conv)

    def _save_conv(self, conv: list[dict]) -> None:
        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        llm = getattr(self.harness, "llm", None)
        checkpoint_enabled = self._canonical_checkpoint_enabled()
        canonical_active = bool(getattr(llm, "persistent_canonical_active", False))
        if checkpoint_enabled and canonical_active:
            llm.sync_canonical_state(conv)

        serialized = json.dumps(conv, indent=2)
        self.session_path.write_text(serialized, encoding="utf-8")

        if not checkpoint_enabled:
            self.canonical_session_path.unlink(missing_ok=True)
            return
        if not canonical_active:
            return
        try:
            checkpoint = llm.export_canonical_state(
                conv,
                conversation_digest=hashlib.sha1(serialized.encode("utf-8")).hexdigest(),
            )
            temporary = self.canonical_session_path.with_name(
                f".{self.canonical_session_path.name}.{os.getpid()}.tmp"
            )
            temporary.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
            os.replace(temporary, self.canonical_session_path)
        except Exception:  # noqa: BLE001 - neutral audit persistence must still survive
            logger.exception("coding canonical checkpoint save failed (non-fatal)")

    def _canonical_checkpoint_enabled(self) -> bool:
        policy = getattr(self.harness, "_policy", None)
        llm = getattr(self.harness, "llm", None)
        return bool(
            getattr(self.harness, "supports_cost_compaction", False)
            and policy is not None
            and policy.enabled
            and policy.enable_native_summary
            and callable(getattr(llm, "export_canonical_state", None))
            and callable(getattr(llm, "import_canonical_state", None))
        )

    def _reset_provider_history(self) -> None:
        reset = getattr(getattr(self.harness, "llm", None), "reset_canonical_history", None)
        if callable(reset):
            reset(preserve_policy_state=True)

    async def build(
        self,
        blueprint: dict,
        *,
        design_board_path: Path | None = None,
        reference_image_paths: list[Path] | None = None,
        max_cost_usd: float | None = None,
    ) -> HarnessResult:
        if max_cost_usd is not None:
            self.harness.max_cost_usd = max_cost_usd
        refs = [p for p in (reference_image_paths or []) if Path(p).exists()]
        text = (
            load_coding_prompt("build.user.md")
            .replace("{blueprint}", json.dumps(blueprint, indent=2))
            .replace("{reference_captions}", reference_caption_block(len(refs)))
        )
        parts: list[dict] = [{"type": "input_text", "text": text}]
        for cap, p in zip(reference_captions(len(refs)), refs):
            parts.append({"type": "input_text", "text": f"{cap} (user reference):"})
            parts.append({"type": "input_image", "image_path": str(p), "detail": "high"})
        if design_board_path and Path(design_board_path).exists():
            parts.append({"type": "input_text", "text": "Design board (generated aid):"})
            parts.append(
                {"type": "input_image", "image_path": str(design_board_path), "detail": "high"}
            )
        msg: object = parts if len(parts) > 1 else text
        res = await self.harness.run(msg, initial_conversation=self._load_conv())
        self._save_conv(res.conversation)
        return res

    def has_history(self) -> bool:
        return self.session_path.exists()

    async def continue_build(
        self,
        *,
        compile_summary: str = "",
        open_items: list[str] | None = None,
        max_cost_usd: float | None = None,
    ) -> HarnessResult:
        if max_cost_usd is not None:
            self.harness.max_cost_usd = max_cost_usd
        items = [str(s).strip() for s in (open_items or []) if str(s).strip()]
        open_block = (
            "\n".join(f"- {s}" for s in items)
            or "(none recorded — probe/compile to see what is still missing)"
        )
        text = (
            load_coding_prompt("build_continue.user.md")
            .replace(
                "{compile_summary}",
                (compile_summary or "(no successful compile recorded yet this build)").strip(),
            )
            .replace("{open_items}", open_block)
        )
        res = await self.harness.run(text, initial_conversation=self._load_conv())
        self._save_conv(res.conversation)
        return res

    async def repair(
        self,
        issues: list[Issue],
        *,
        evidence_images: list[Path] | None = None,
        review_round: int | None = None,
        max_cost_usd: float | None = None,
    ) -> HarnessResult:
        if max_cost_usd is not None:
            self.harness.max_cost_usd = max_cost_usd
        issues_text = "\n\n".join(_format_issue(i) for i in issues)
        text = load_coding_prompt("repair.user.md").replace("{issues}", issues_text)
        imgs = [p for p in (evidence_images or []) if Path(p).exists()]
        if imgs:
            parts: list[dict] = [
                {"type": "input_text", "text": text},
                {
                    "type": "input_text",
                    "text": "ReviewAgent renders of your CURRENT build showing the issues above:",
                },
            ]
            for p in imgs:
                image_part: dict[str, object] = {
                    "type": "input_image",
                    "image_path": str(p),
                    "detail": "high",
                }
                if isinstance(review_round, int) and review_round > 0:
                    image_part["_meta"] = {"review_evidence_round": review_round}
                parts.append(image_part)
            msg: object = parts
        else:
            msg = text
        res = await self.harness.run(
            msg,
            initial_conversation=self._load_conv(),
            entry_reason="review_repair",
        )
        self._save_conv(res.conversation)
        return res
