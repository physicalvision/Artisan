from __future__ import annotations

from typing import Any


def reference_captions(n: int) -> list[str]:
    return [f"Reference {i + 1}" for i in range(n)]


def reference_caption_block(n: int) -> str:
    if n <= 0:
        return "No reference images are attached."
    lines = "\n".join(f"- {c}" for c in reference_captions(n))
    return f"Attached reference images (in order):\n{lines}"


def reference_observations_summary(ro: Any) -> str:
    if ro is None:
        return ""
    if isinstance(ro, str):
        return ro
    if not isinstance(ro, dict):
        return str(ro)

    bits: list[str] = []
    interp = ro.get("multi_image_interpretation")
    if interp:
        bits.append(f"interpretation={interp}")

    per = ro.get("per_reference")
    if isinstance(per, list) and per:
        bits.append(f"{len(per)} reference(s)")

    motion = ro.get("motion_evidence")
    if isinstance(motion, list) and motion:
        parts = []
        for m in motion:
            if not isinstance(m, dict):
                continue
            name = m.get("moving_part", "?")
            rng = m.get("candidate_range_hint")
            parts.append(f"{name} ({rng})" if rng else str(name))
        if parts:
            bits.append("observed motion: " + "; ".join(parts))

    return " | ".join(bits)
