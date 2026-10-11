from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class RenderDevicePolicy:
    mode: str = "cpu"
    backend: str = "OPTIX"
    gpu_index: int = 0

    def __post_init__(self) -> None:
        if self.mode not in {"cpu", "gpu", "auto"}:
            raise ValueError("ARTISAN_RENDER_DEVICE must be cpu, gpu, or auto")
        if self.backend not in {"OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"}:
            raise ValueError("Unsupported ARTISAN_RENDER_GPU_BACKEND")
        if self.gpu_index < 0:
            raise ValueError("ARTISAN_RENDER_GPU_INDEX must be nonnegative")

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> RenderDevicePolicy:
        values = os.environ if environ is None else environ
        return cls(
            mode=values.get("ARTISAN_RENDER_DEVICE", "cpu").strip().lower(),
            backend=values.get("ARTISAN_RENDER_GPU_BACKEND", "OPTIX").strip().upper(),
            gpu_index=int(values.get("ARTISAN_RENDER_GPU_INDEX", "0")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def configure_cycles_device(bpy: Any, policy: RenderDevicePolicy) -> dict[str, Any]:
    result: dict[str, Any] = {"requested": policy.to_dict(), "actual": "CPU"}
    if policy.mode == "cpu":
        bpy.context.scene.cycles.device = "CPU"
        return result
    try:
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = policy.backend
        preferences.get_devices()
        available = sorted(
            (device for device in preferences.devices if device.type == policy.backend),
            key=lambda device: device.id,
        )
        if not available:
            raise RuntimeError(f"No {policy.backend} rendering device is available")
    except (RuntimeError, TypeError) as exc:
        if policy.mode != "auto":
            raise
        bpy.context.scene.cycles.device = "CPU"
        result["fallback_reason"] = str(exc)
        return result
    if policy.gpu_index >= len(available):
        raise ValueError(
            f"GPU index {policy.gpu_index} is outside the {len(available)} visible {policy.backend} devices"
        )
    selected = available[policy.gpu_index]
    for device in preferences.devices:
        device.use = device.id == selected.id and device.type == policy.backend
    bpy.context.scene.cycles.device = "GPU"
    result.update(actual=policy.backend, device_id=selected.id, device_name=selected.name)
    return result
