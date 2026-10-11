
from __future__ import annotations

import base64
import binascii
import math
import re
from pathlib import Path
from typing import Any

from pipeline.llm._image_tokens import sniff_image_dimensions
from pipeline.llm.errors import CostAccountingError, ProviderConfigurationError
from pipeline.llm.model_registry import image_tool_pricing

_CHARS_PER_TEXT_TOKEN = 4
_GPT_IMAGE_2_QUALITY_GRID = {"low": 16, "medium": 48, "high": 96}
_GPT_IMAGE_2_5_QUALITY_GRID = {"low": 16, "medium": 24, "high": 48, "xhigh": 64, "max": 96}
_OUTPUT_TOKEN_GRIDS = {
    "gpt-image-2": _GPT_IMAGE_2_QUALITY_GRID,
    "gpt-image-2.5-sunburst": _GPT_IMAGE_2_5_QUALITY_GRID,
    "gpt-image-2.5-sunburst-2026-09-08": _GPT_IMAGE_2_5_QUALITY_GRID,
}
_SIZE_RE = re.compile(r"^(\d+)x(\d+)$")

PRICING_SOURCE = "https://developers.openai.com/api/docs/pricing#image-generation"
TOKEN_SOURCE = "https://developers.openai.com/api/docs/guides/image-generation#cost-and-latency"
VISION_SOURCE = "https://developers.openai.com/api/docs/guides/images-vision#calculating-costs"


def estimate_gpt_image_2_output_tokens(width: int, height: int, quality: str) -> int | None:
    return estimate_image_output_tokens(width, height, quality, "gpt-image-2")


def estimate_image_output_tokens(
    width: int,
    height: int,
    quality: str,
    model: str,
    *,
    verified_output_size: tuple[int, int] | None = None,
) -> int | None:
    grid = _OUTPUT_TOKEN_GRIDS.get(model, {}).get(str(quality).lower())
    if (
        width <= 0
        or height <= 0
        or grid is None
        or ((width % 16 or height % 16) and verified_output_size != (width, height))
        or max(width, height) > 3840
        or max(width, height) > 3 * min(width, height)
        or not 655_360 <= width * height <= 8_294_400
    ):
        return None
    long_edge = max(width, height)
    short_edge = min(width, height)
    short_grid = round(grid / (long_edge / short_edge))
    grid_width, grid_height = (grid, short_grid) if width >= height else (short_grid, grid)
    latent_tokens = grid_width * grid_height
    return math.ceil(latent_tokens * (2_000_000 + width * height) / 4_000_000)


def _parse_size(value: Any) -> tuple[int, int] | None:
    match = _SIZE_RE.fullmatch(str(value or "").strip().lower())
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _output_image_dimensions(output_item: dict[str, Any]) -> tuple[int, int] | None:
    result = output_item.get("result")
    if not isinstance(result, str) or not result:
        return None
    try:
        return sniff_image_dimensions(base64.b64decode(result, validate=True)[:131_072])
    except (ValueError, binascii.Error):
        return None


def _estimate_high_fidelity_input_tokens(width: int, height: int) -> int:
    scale = min(1.0, 2048 / max(width, height))
    scaled_width = width * scale
    scaled_height = height * scale
    short_edge = min(scaled_width, scaled_height)
    if short_edge > 0:
        detail_scale = 512 / short_edge
        scaled_width *= detail_scale
        scaled_height *= detail_scale
    tiles = math.ceil(scaled_width / 512) * math.ceil(scaled_height / 512)
    supplement = 4160 if width == height else 6240
    return 65 + 129 * tiles + supplement


def _estimate_input_image_tokens(image_paths: list[str]) -> tuple[int, list[str]]:
    total = 0
    unknown: list[str] = []
    for raw_path in image_paths:
        path = Path(raw_path)
        try:
            with path.open("rb") as fp:
                dimensions = sniff_image_dimensions(fp.read(131_072))
        except OSError:
            dimensions = None
        if dimensions is None:
            unknown.append(str(path))
            continue
        total += _estimate_high_fidelity_input_tokens(*dimensions)
    return total, unknown


def _reported_image_tokens(output_item: dict[str, Any]) -> dict[str, int] | None:
    usage = output_item.get("usage")
    if usage is None:
        return None
    if not isinstance(usage, dict):
        raise CostAccountingError("Image tool usage must be an object")
    details = usage.get("input_tokens_details")
    output_details = usage.get("output_tokens_details")
    if not isinstance(details, dict):
        raise CostAccountingError("Image tool usage lacks input modality counts")
    if output_details is not None and not isinstance(output_details, dict):
        raise CostAccountingError("Image tool output modality counts must be an object")
    if any(
        type(usage.get(key)) is not int or usage[key] < 0
        for key in ("input_tokens", "output_tokens")
    ):
        raise CostAccountingError("Image tool usage contains invalid total token counts")
    tokens = {
        "text_input": details.get("text_tokens"),
        "image_input": details.get("image_tokens"),
        "image_output": (
            output_details.get("image_tokens")
            if isinstance(output_details, dict)
            else usage.get("output_tokens")
        ),
    }
    if any(type(value) is not int or value < 0 for value in tokens.values()):
        raise CostAccountingError("Image tool usage contains missing or invalid token counts")
    if tokens["image_output"] == 0:
        raise CostAccountingError("A generated image has no reported output tokens")
    if tokens["image_output"] > usage["output_tokens"]:
        raise CostAccountingError("Image tool output modality exceeds its total token count")
    if usage.get("input_tokens") != tokens["text_input"] + tokens["image_input"]:
        raise CostAccountingError("Image tool input token totals do not match modality counts")
    return tokens


def estimate_image_generation_cost(
    *,
    output_item: dict[str, Any],
    original_prompt: str,
    image_paths: list[str],
    pricing_model: str,
    model_source: str,
    partial_images: int = 0,
) -> dict[str, Any]:
    if type(partial_images) is not int or partial_images < 0:
        raise CostAccountingError("Partial image count must be a nonnegative integer")
    model = str(pricing_model or "").strip().lower()
    unknown_components: list[str] = []
    assumptions = ["cached image-generation inputs are charged at uncached rates"]
    try:
        rates = image_tool_pricing(model)
    except ProviderConfigurationError:
        rates = None
    if rates is None:
        return {
            "billing_status": "unpriced",
            "model": model or None,
            "model_source": model_source,
            "estimated_cost_usd": None,
            "unknown_components": ["unsupported_image_pricing_model"],
            "pricing_source": PRICING_SOURCE,
            "token_source": TOKEN_SOURCE,
        }

    dimensions = _parse_size(output_item.get("size"))
    quality = str(output_item.get("quality") or "").lower()
    reported = _reported_image_tokens(output_item)
    unknown_images: list[str] = []
    if reported is not None:
        text_input_tokens = reported["text_input"]
        image_input_tokens = reported["image_input"]
        output_tokens = reported["image_output"]
        assumptions.append("reported tool usage takes precedence over local token estimates")
    else:
        revised_prompt = output_item.get("revised_prompt")
        priced_prompt = revised_prompt if isinstance(revised_prompt, str) else original_prompt
        text_input_tokens = max(1, math.ceil(len(priced_prompt) / _CHARS_PER_TEXT_TOKEN))
        image_input_tokens, unknown_images = _estimate_input_image_tokens(image_paths)
        if unknown_images:
            unknown_components.append("input_image_tokens")
        actual_dimensions = _output_image_dimensions(output_item)
        if actual_dimensions is not None and dimensions != actual_dimensions:
            unknown_components.append("output_image_dimensions")
        output_tokens = (
            estimate_image_output_tokens(
                *dimensions, quality, model, verified_output_size=actual_dimensions
            )
            if dimensions is not None
            else None
        )
        if dimensions is not None and (dimensions[0] % 16 or dimensions[1] % 16):
            assumptions.append(
                "automatic output dimensions are verified against image bytes; the calculator "
                "formula is extrapolated beyond its explicit-request size alignment domain"
            )
        if output_tokens is not None:
            output_tokens += partial_images * 100
        assumptions.extend(
            [
                "text input tokens estimated as ceil(characters / 4)",
                "image input tokens extrapolated from GPT Image 1 high-fidelity sizing; "
                "this is not an official input-token formula for the selected model",
                "image output tokens use the selected model's official calculator grid",
                "each observed partial image adds 100 output tokens",
            ]
        )
    if output_tokens is None:
        unknown_components.append("image_output_tokens")
    component_costs: dict[str, float | None] = {
        "text_input": text_input_tokens * rates["text_input"] / 1_000_000,
        "image_input": image_input_tokens * rates["image_input"] / 1_000_000,
        "image_output": (
            output_tokens * rates["image_output"] / 1_000_000 if output_tokens is not None else None
        ),
    }
    estimated_cost = (
        sum(value for value in component_costs.values() if value is not None)
        if not unknown_components
        else None
    )
    status = "reported_usage_upper_bound" if reported is not None else "budget_estimate"
    if unknown_components:
        status = "unpriced"
    return {
        "billing_status": status,
        "model": model,
        "model_source": model_source,
        "quality": quality or None,
        "size": output_item.get("size"),
        "estimated_tokens": {
            "text_input": text_input_tokens,
            "image_input": image_input_tokens,
            "image_output": output_tokens,
        },
        "rates_usd_per_million_tokens": rates,
        "costs_usd": {
            key: (round(value, 8) if value is not None else None)
            for key, value in component_costs.items()
        },
        "estimated_cost_usd": round(estimated_cost, 8) if estimated_cost is not None else None,
        "reported_tokens": reported,
        "unknown_components": unknown_components,
        "unknown_image_paths": unknown_images,
        "assumptions": assumptions,
        "pricing_source": PRICING_SOURCE,
        "token_source": TOKEN_SOURCE,
        "vision_source": VISION_SOURCE,
    }
