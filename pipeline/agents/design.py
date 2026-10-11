from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import time
import uuid
from pathlib import Path

import httpx

from pipeline.agents._references import reference_caption_block, reference_captions
from pipeline.agents.design_harness import DesignBlueprintHarness, DesignResult
from pipeline.agents.prompts.design import load_design_prompt
from pipeline.contracts import validate_support_mode
from pipeline.cost import calculate_cost, pricing_for_provider_model
from pipeline.image_billing import estimate_image_generation_cost
from pipeline.llm.credentials import image_api_key_from_env
from pipeline.llm.endpoints import official_openai_base_url
from pipeline.llm.errors import CostAccountingError, ProviderConfigurationError
from pipeline.llm.model_registry import image_models_from_env, validate_model
from pipeline.llm.openai_codec import extract_usage
from pipeline.timing import _null_measure

logger = logging.getLogger(__name__)


def _image_base_url() -> str:
    return official_openai_base_url(
        os.environ.get("ARTISAN_IMAGE_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    )


def _image_api_key() -> str | None:
    return image_api_key_from_env()


def _official_responses_url() -> str:
    return _image_base_url() + "/responses"


def _official_headers(key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "X-Client-Request-Id": str(uuid.uuid4()),
    }


def _append_event(seq: list[str], t: str) -> None:
    if seq:
        prev = seq[-1]
        head, _, mult = prev.partition("×")
        if head == t:
            n = int(mult) if mult else 1
            seq[-1] = f"{t}×{n + 1}"
            return
    seq.append(t)


def _data_url(image_path: str) -> str:
    p = Path(image_path).expanduser()
    mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def _image_request_body(prompt: str, image_paths: list[str] | None = None) -> dict:
    image_model, image_tool_model = image_models_from_env()
    content: list[dict] = [{"type": "input_text", "text": prompt}]
    for image_path in image_paths or []:
        if image_path and Path(image_path).exists():
            content.append(
                {"type": "input_image", "image_url": _data_url(image_path), "detail": "high"}
            )
    instructions = (
        "You render design-board reference imagery for an articulated-object engineering pipeline."
    )
    image_tool: dict = {"type": "image_generation", "model": image_tool_model}
    return {
        "model": image_model,
        "instructions": instructions,
        "input": [{"type": "message", "role": "user", "content": content}],
        "tools": [image_tool],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "store": False,
        "stream": True,
    }


def _generate_image(
    prompt: str, out_png: Path, *, image_paths: list[str] | None = None, retries: int = 2
) -> tuple[bool, dict]:
    attempts: list[dict] = []
    last_summary = ""
    wire = "official"
    key = _image_api_key()
    if not key:
        raise ProviderConfigurationError(
            "Image credentials are required. Set ARTISAN_IMAGE_API_KEY, OPENAI_API_KEY, or OPENAI_API_KEYS."
        )
    url = _official_responses_url()
    body = _image_request_body(prompt, image_paths)
    for attempt in range(1, retries + 1):
        got = None
        final_response: dict | None = None
        request_id: str | None = None
        event_counts: dict[str, int] = {}
        event_sequence: list[str] = []
        last_event_type: str | None = None
        request_accepted = False
        try:
            request_headers = _official_headers(key)
            with httpx.stream("POST", url, headers=request_headers, json=body, timeout=600) as r:
                request_id = r.headers.get("x-request-id") or r.headers.get("openai-request-id")
                if r.status_code != 200:
                    last_summary = f"status {r.status_code}: {r.read()[:200]!r}"
                    attempts.append(
                        {
                            "attempt": attempt,
                            "summary": last_summary,
                            "last_event_type": None,
                            "event_counts": {},
                            "event_sequence": [],
                        }
                    )
                    if r.status_code in (500, 502, 503, 504) and attempt < retries:
                        time.sleep(3 * attempt)
                        continue
                    return False, {
                        "summary": last_summary[-400:],
                        "attempts": attempts,
                        "wire": wire,
                    }
                request_accepted = True
                for line in r.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    try:
                        ev = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue
                    t = ev.get("type", "")
                    last_event_type = t
                    event_counts[t] = event_counts.get(t, 0) + 1
                    _append_event(event_sequence, t)
                    if "image_generation_call" in t and ev.get("partial_image_b64"):
                        got = ev["partial_image_b64"]
                    elif t == "response.completed":
                        response = ev.get("response", {})
                        final_response = response if isinstance(response, dict) else None
                        got = None
                        for o in response.get("output", []):
                            if o.get("type") == "image_generation_call" and o.get("result"):
                                got = o["result"]
            if got and final_response is not None:
                out_png.parent.mkdir(parents=True, exist_ok=True)
                out_png.write_bytes(base64.b64decode(got))
                if out_png.exists() and out_png.stat().st_size > 2000:
                    last_summary = f"ok (attempt {attempt}, {out_png.stat().st_size}B)"
                    accounting = _image_response_accounting(
                        final_response,
                        prompt=prompt,
                        image_paths=image_paths or [],
                        request_id=request_id,
                        requested_model=body["model"],
                        requested_tool_model=body["tools"][0]["model"],
                        partial_images=sum(
                            count
                            for event, count in event_counts.items()
                            if "image_generation_call" in event and event.endswith(".partial_image")
                        ),
                    )
                    attempts.append(
                        {
                            "attempt": attempt,
                            "summary": last_summary,
                            "last_event_type": last_event_type,
                            "event_counts": event_counts,
                            "event_sequence": event_sequence,
                            "request_id": request_id,
                        }
                    )
                    return True, {
                        "summary": last_summary,
                        "attempts": attempts,
                        "accounting": accounting,
                        "wire": wire,
                    }
            raise CostAccountingError(
                "Accepted image request produced no completed image with verifiable billing; "
                "automatic retry would risk an unrecorded duplicate charge"
            )
        except (ProviderConfigurationError, CostAccountingError):
            raise
        except Exception as exc:  # noqa: BLE001
            if request_accepted:
                raise CostAccountingError(
                    "Accepted image request failed before billing could be recorded"
                ) from exc
            last_summary = f"{type(exc).__name__}: {exc}"
        attempts.append(
            {
                "attempt": attempt,
                "summary": last_summary,
                "last_event_type": last_event_type,
                "event_counts": event_counts,
                "event_sequence": event_sequence,
            }
        )
        if attempt < retries:
            time.sleep(3 * attempt)
            continue
        return False, {
            "summary": last_summary[-400:],
            "attempts": attempts,
            "wire": wire,
        }
    return False, {"summary": last_summary[-400:], "attempts": attempts, "wire": wire}


def _image_response_accounting(
    response: dict | None,
    *,
    prompt: str,
    image_paths: list[str],
    request_id: str | None,
    requested_model: str | None = None,
    requested_tool_model: str | None = None,
    partial_images: int = 0,
) -> dict:
    response = response or {}
    if requested_model is None or requested_tool_model is None:
        configured_model, configured_tool_model = image_models_from_env()
        requested_model = requested_model or configured_model
        requested_tool_model = requested_tool_model or configured_tool_model
    mainline_model = str(response.get("model") or requested_model)
    raw_usage = response.get("usage")
    if not isinstance(raw_usage, dict) or any(
        type(raw_usage.get(key)) is not int or raw_usage[key] < 0
        for key in ("input_tokens", "output_tokens")
    ):
        raise CostAccountingError("Image response lacks valid mainline token usage")
    mainline_usage = extract_usage(response)
    try:
        validate_model(mainline_model, "image_envelope")
    except ProviderConfigurationError as exc:
        raise CostAccountingError("Image response envelope model cannot be priced") from exc
    pricing = pricing_for_provider_model("openai", mainline_model)
    if mainline_usage is None or pricing is None:
        raise CostAccountingError("Image response mainline charge cannot be calculated")
    mainline_cost_usd = calculate_cost(mainline_usage, pricing).total_cost
    image_items = [
        item
        for item in response.get("output", [])
        if isinstance(item, dict) and item.get("type") == "image_generation_call"
    ]
    if not image_items:
        raise CostAccountingError("Image response contains no image-generation call to price")
    image_estimates = []
    for index, image_item in enumerate(image_items):
        reported_image_model = image_item.get("model")
        pricing_model = str(reported_image_model or requested_tool_model)
        try:
            validate_model(pricing_model, "image_tool")
        except ProviderConfigurationError as exc:
            raise CostAccountingError("Image response tool model cannot be priced") from exc
        image_estimate = estimate_image_generation_cost(
            output_item=image_item,
            original_prompt=prompt,
            image_paths=image_paths,
            pricing_model=pricing_model,
            model_source="response" if reported_image_model else "configured_tool_model",
            partial_images=partial_images if index == 0 else 0,
        )
        if image_estimate.get("estimated_cost_usd") is None or image_estimate.get(
            "unknown_components"
        ):
            raise CostAccountingError(
                "Image response charge is incomplete: "
                + ", ".join(image_estimate.get("unknown_components") or ["missing image cost"])
            )
        image_estimates.append(image_estimate)
    image_cost = sum(item["estimated_cost_usd"] for item in image_estimates)
    return {
        "billing_status": "budget_estimate",
        "request_id": request_id,
        "response_id": response.get("id"),
        "mainline_model": mainline_model,
        "mainline_usage": mainline_usage,
        "exact_mainline_cost_usd": round(mainline_cost_usd, 8),
        "image_generation": image_estimates[0],
        "image_generation_calls": image_estimates,
        "estimated_image_cost_usd": image_cost,
        "budget_cost_usd": round(mainline_cost_usd + image_cost, 8),
        "cost_basis": "exact_mainline_plus_estimated_image",
        "unknown_components": [],
    }


class DesignAgent:
    def __init__(
        self,
        *,
        blueprint_path: Path | None = None,
        session_path: Path | None = None,
        selfcheck_report_path: Path | None = None,
        trace_path: Path | None = None,
        max_cost_usd: float | None = 10.0,
        cassette: object | None = None,
        recorder: object | None = None,
        harness: DesignBlueprintHarness | None = None,
    ) -> None:
        from pipeline.providers import pipeline_model, pipeline_provider, pipeline_thinking

        self.provider = pipeline_provider("design")
        self.model_id = pipeline_model("design")
        self.thinking_level = pipeline_thinking("design")
        self.max_cost_usd = max_cost_usd
        self.cassette = cassette
        self.trace_path = trace_path
        self.system_prompt = load_design_prompt("system.md")
        self._blueprint_path = blueprint_path
        self._session_path = session_path
        self._selfcheck_report_path = selfcheck_report_path
        self._recorder = recorder
        self.recorder = recorder.for_agent("design") if recorder is not None else None
        self._harness = harness
        self._image_budget_cost_usd = 0.0

    @property
    def cost_usd(self) -> float:
        harness = getattr(self, "_harness", None)
        harness_cost = harness.cost_usd if harness is not None else 0.0
        return harness_cost + getattr(self, "_image_budget_cost_usd", 0.0)

    def _replaying(self) -> bool:
        return bool(getattr(getattr(self, "cassette", None), "replaying", False))

    def _get_harness(self, bp_path: Path) -> DesignBlueprintHarness:
        if self._blueprint_path is not None and self._blueprint_path != bp_path:
            raise ValueError("DesignAgent received a blueprint path different from its workspace")
        if self._harness is not None:
            if self._harness.blueprint_path != bp_path:
                raise ValueError("DesignAgent cannot switch blueprint paths within one session")
            return self._harness
        session_path = self._session_path or bp_path.parent / "sessions" / "design.json"
        report_path = (
            self._selfcheck_report_path or bp_path.parent / "reports" / "P1_selfcheck.json"
        )
        self._harness = DesignBlueprintHarness(
            bp_path,
            session_path=session_path,
            selfcheck_report_path=report_path,
            provider=self.provider,
            model_id=self.model_id,
            thinking_level=self.thinking_level,
            max_cost_usd=self.max_cost_usd,
            trace_path=self.trace_path,
            cassette=self.cassette,
            recorder=self._recorder,
        )
        return self._harness

    async def intake(
        self, prompt: str, bp_path: Path, *, image_paths: list[str] | None = None
    ) -> DesignResult:
        imgs = [Path(p) for p in (image_paths or []) if Path(p).exists()]
        user_text = (
            load_design_prompt("intake.user.md")
            .replace("{prompt}", prompt)
            .replace("{reference_captions}", reference_caption_block(len(imgs)))
        )
        if imgs:
            message: object = [{"type": "input_text", "text": user_text}]
            for caption, image_path in zip(reference_captions(len(imgs)), imgs):
                message.append({"type": "input_text", "text": f"{caption}:"})
                message.append(
                    {"type": "input_image", "image_path": str(image_path), "detail": "high"}
                )
        else:
            message = user_text
        return await self._get_harness(bp_path).run_intake(message)

    async def blueprint(
        self,
        prompt: str,
        brief: dict,
        bp_path: Path,
        *,
        image_paths: list[str] | None = None,
        reopen: bool = False,
    ) -> DesignResult:
        validate_support_mode(brief)
        current = (
            json.loads(bp_path.read_text(encoding="utf-8"))
            if bp_path.exists()
            else {"asset_brief": brief}
        )
        template = load_design_prompt("blueprint.template.json")
        imgs = [Path(p) for p in (image_paths or []) if Path(p).exists()]
        user_text = (
            load_design_prompt("blueprint.user.md")
            .replace("{prompt}", prompt)
            .replace("{current}", json.dumps(current, indent=2))
            .replace("{template}", template)
            .replace("{reference_captions}", reference_caption_block(len(imgs)))
        )
        return await self._get_harness(bp_path).run_blueprint(
            user_text, asset_brief=brief, reopen=reopen
        )

    @staticmethod
    def _axis_phrase(joint: dict) -> str:
        verb = "along" if joint.get("type") == "prismatic" else "about"
        axis = joint.get("axis") or []
        named = {
            (1.0, 0.0, 0.0): "the left-right (X) axis",
            (0.0, 1.0, 0.0): "the front-back (Y) axis",
            (0.0, 0.0, 1.0): "the vertical (Z) axis",
        }
        key = tuple(abs(float(a)) for a in axis) if len(axis) == 3 else None
        desc = named.get(key, f"axis {axis}")
        return f"{verb} {desc} on the hinge/rail line"

    @staticmethod
    def _board_slots(brief: dict, bp: dict) -> dict:
        part_objs = bp.get("parts", [])
        parts_panel = "; ".join(
            f"{p.get('name', '?')} ({p.get('role', 'part')}"
            + (f", {p['geometry_hint']}" if p.get("geometry_hint") else "")
            + ")"
            for p in part_objs
        )
        moving = {"revolute", "continuous", "prismatic"}
        articulation = "; ".join(
            f"{j.get('name', '?')}: {j.get('type')} of {j.get('child', '?')} "
            f"{DesignAgent._axis_phrase(j)}, closed→open"
            for j in bp.get("joints", [])
            if j.get("type") in moving
        )
        mats = sorted({p.get("material") for p in part_objs if p.get("material")})
        materials = ", ".join(m for m in mats if m) or "realistic material finishes"
        td = bp.get("target_dimensions", {})
        if all(k in td for k in ("x", "y", "z")):
            key_proportions = f"{td['x']} x {td['y']} x {td['z']} m"
        elif td:
            key_proportions = json.dumps(td)
        else:
            key_proportions = ""
        name = brief.get("name") or "articulated object"
        category = brief.get("category") or ""
        return {
            "subject": f"{name} ({category})" if category else name,
            "parts_panel": parts_panel,
            "articulation": articulation or "as described in the blueprint",
            "materials": materials,
            "palette": "true material colours under soft, even studio lighting on a neutral backdrop",
            "key_proportions": key_proportions,
            "avoid_extra": "",
        }

    @staticmethod
    def _board_prompt(slots: dict) -> str:
        avoid_extra = slots.get("avoid_extra") or ""
        if avoid_extra:
            avoid_extra = f"; {avoid_extra}"
        repl = {
            "{subject}": slots.get("subject", "an articulated object"),
            "{parts_panel}": slots.get("parts_panel", ""),
            "{articulation}": slots.get("articulation", "as described"),
            "{palette}": slots.get("palette", "neutral light background, muted material tones"),
            "{materials}": slots.get("materials", "neutral"),
            "{key_proportions}": slots.get("key_proportions", ""),
            "{avoid_extra}": avoid_extra,
        }
        out = load_design_prompt("design_board.skeleton.md")
        for k, v in repl.items():
            out = out.replace(k, str(v))
        return out.rstrip("\n")

    async def generate_design_board(
        self,
        brief: dict,
        bp: dict,
        png: Path,
        md: Path,
        meta: Path,
        *,
        image_paths: list[str] | None = None,
    ) -> dict:
        png.parent.mkdir(parents=True, exist_ok=True)
        slots = self._board_slots(brief, bp)
        prompt = self._board_prompt(slots)
        parts = [p.get("name", "?") for p in bp.get("parts", [])]
        imgs = [p for p in (image_paths or []) if Path(p).exists()]

        status, mode = "skipped", "none"
        if self._replaying():
            if png.exists() and png.stat().st_size > 2000:
                status, mode = "available", "replay (reused artifact)"
            detail = {"summary": f"replay: image request skipped ({mode})"}
        else:
            _rec = self.recorder
            with _rec.measure() if _rec is not None else _null_measure() as _img_s:
                ok, detail = _generate_image(prompt, png, image_paths=imgs)
            accounting = detail.get("accounting") if isinstance(detail, dict) else None
            budget_cost = (
                accounting.get("budget_cost_usd") if isinstance(accounting, dict) else None
            )
            if ok and isinstance(budget_cost, (int, float)):
                self._image_budget_cost_usd += float(budget_cost)
            if _rec is not None:
                _rec.record(
                    kind="llm",
                    name="design_board_image",
                    sample=_img_s,
                    ok=ok,
                    tokens=(accounting or {}).get("mainline_usage"),
                    cost_usd=(
                        float(budget_cost) if isinstance(budget_cost, (int, float)) else None
                    ),
                    billing_status=(accounting or {}).get("billing_status"),
                    cost_basis=(accounting or {}).get("cost_basis"),
                    cost_components=(accounting or {}),
                    expected_cold=False,
                )
            if ok and png.exists() and png.stat().st_size > 2000:
                status = "available"
                mode = "official OpenAI responses stream"

        m = {
            "status": status,
            "mode": mode,
            "prompt": prompt,
            "slots": slots,
            "png": str(png) if status == "available" else None,
            "md": None,
            "reference_images_used": len(imgs),
            "intended_panels": parts,
            "detail": detail,
        }
        meta.write_text(json.dumps(m, indent=2), encoding="utf-8")
        return m

    def reconcile(self, bp_path: Path, stage: str, changes: list[str]) -> dict:
        bp = json.loads(bp_path.read_text(encoding="utf-8"))
        bp.setdefault("version_history", []).append(
            {"version": f"v_{stage}", "stage": stage, "changes": changes}
        )
        bp_path.write_text(json.dumps(bp, indent=2), encoding="utf-8")
        return bp
