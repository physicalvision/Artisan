
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from pipeline.articulated_glb import LINK_NODE_PREFIX, VISUAL_NODE_PREFIX, read_glb_json

BAKE_POLICY_VERSION = "articulated-pbr-bake-v1"
_TEXTURE_FILENAMES = ("color.png", "roughness.png", "metalness.png", "normal.png")


class ArticulatedBakeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BakePolicy:
    texels_per_meter: int = 1024
    fallback_texels_per_meter: int = 512
    minimum_texels_per_meter: int = 256
    min_atlas_size: int = 1024
    max_atlas_size: int = 4096
    padding_2k: int = 8
    soft_size_bytes: int = 64 * 1024 * 1024
    hard_size_bytes: int = 128 * 1024 * 1024
    timeout_seconds: int = 900
    packing_efficiency: float = 0.72
    projection_scale: float = 3.0

    @classmethod
    def from_env(cls) -> "BakePolicy":
        defaults = cls()
        policy = cls(
            texels_per_meter=_positive_int_env(
                "ARTISAN_PBR_TEXELS_PER_METER", defaults.texels_per_meter
            ),
            fallback_texels_per_meter=_positive_int_env(
                "ARTISAN_PBR_FALLBACK_TEXELS_PER_METER",
                defaults.fallback_texels_per_meter,
            ),
            minimum_texels_per_meter=_positive_int_env(
                "ARTISAN_PBR_MINIMUM_TEXELS_PER_METER",
                defaults.minimum_texels_per_meter,
            ),
            min_atlas_size=_positive_int_env(
                "ARTISAN_PBR_MIN_ATLAS_SIZE", defaults.min_atlas_size
            ),
            max_atlas_size=_positive_int_env(
                "ARTISAN_PBR_MAX_ATLAS_SIZE", defaults.max_atlas_size
            ),
            timeout_seconds=_positive_int_env(
                "ARTISAN_PBR_BAKE_TIMEOUT_SECONDS", defaults.timeout_seconds
            ),
        )
        for name, value in (
            ("min_atlas_size", policy.min_atlas_size),
            ("max_atlas_size", policy.max_atlas_size),
        ):
            if value & (value - 1):
                raise ValueError(f"{name} must be a power of two (got {value})")
        if policy.min_atlas_size > policy.max_atlas_size:
            raise ValueError("min_atlas_size must not exceed max_atlas_size")
        if policy.fallback_texels_per_meter > policy.texels_per_meter:
            raise ValueError("fallback texel density must not exceed primary density")
        if policy.minimum_texels_per_meter > policy.fallback_texels_per_meter:
            raise ValueError("minimum texel density must not exceed fallback density")
        return policy

    def driver_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["padding"] = max(
            2,
            round(self.padding_2k * self.max_atlas_size / 2048),
        )
        return payload


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive (got {value})")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def _directory_identity(root_text: str) -> str:
    root = Path(root_text)
    rows: list[dict[str, object]] = []
    if root.is_dir():
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": _sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@lru_cache(maxsize=8)
def _binary_identity(path_text: str) -> str:
    path = Path(path_text)
    if not path.is_file():
        return "unavailable"
    return _sha256_file(path)


def bake_dependency_identity(texture_root: Path | None = None) -> dict[str, str]:
    from pipeline.blender.runtime import blender_bin

    root = Path(texture_root or Path(__file__).parent / "assets" / "textures")
    blender = blender_bin()
    policy_payload = BakePolicy.from_env().driver_payload()
    return {
        "policy_version": BAKE_POLICY_VERSION,
        "policy_sha256": hashlib.sha256(
            json.dumps(policy_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "texture_library_sha256": _directory_identity(str(root.absolute())),
        "blender_binary_sha256": _binary_identity(str(Path(blender).absolute())),
    }


def collect_texture_bake_specs(
    renderer_specs: dict[str, dict[str, Any]],
    source_specs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for material_name, renderer_spec in sorted(renderer_specs.items()):
        texture_dir_value = renderer_spec.get("texture_dir")
        if not texture_dir_value:
            continue
        texture_dir = Path(str(texture_dir_value))
        color_path = texture_dir / "color.png"
        if not color_path.is_file():
            continue
        maps = {
            filename.removesuffix(".png"): str(path.absolute())
            for filename in _TEXTURE_FILENAMES
            if (path := texture_dir / filename).is_file()
        }
        source = dict(source_specs.get(material_name) or {})
        selected[material_name] = {
            **source,
            "texture_dir": str(texture_dir.absolute()),
            "maps": maps,
        }
    return selected


def _run_blender_bake(
    source_glb: Path,
    material_specs: dict[str, dict[str, Any]],
    temp_dir: Path,
    policy: BakePolicy,
) -> tuple[Path, dict[str, Any], str]:
    from pipeline.blender.runtime import blender_bin

    attempt_dir = temp_dir / f"tpm-{policy.texels_per_meter}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    output_glb = attempt_dir / "object.articulated.baked.glb"
    result_path = attempt_dir / "bake_result.json"
    spec_path = attempt_dir / "bake_spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "schema_version": "artisan-articulated-bake-spec-v1",
                "source_glb": str(source_glb.absolute()),
                "output_glb": str(output_glb.absolute()),
                "result_json": str(result_path.absolute()),
                "atlas_dir": str(attempt_dir.absolute()),
                "policy": policy.driver_payload(),
                "materials": material_specs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    driver = Path(__file__).with_name("_articulated_bake_driver.py")
    cmd = [
        blender_bin(),
        "--background",
        "--factory-startup",
        "--python",
        str(driver),
        "--",
        str(spec_path),
    ]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=policy.timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ArticulatedBakeError(f"Blender texture bake could not run: {exc}") from exc
    result: dict[str, Any] = {}
    if result_path.is_file():
        try:
            loaded = json.loads(result_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                result = loaded
        except (OSError, ValueError):
            result = {}
    if completed.returncode != 0 or result.get("status") != "ok" or not output_glb.is_file():
        detail = str(result.get("error") or "Blender did not produce a validated result")
        tail = "\n".join((completed.stdout + "\n" + completed.stderr).splitlines()[-20:])
        raise ArticulatedBakeError(f"Blender texture bake failed: {detail}\n{tail}")
    return output_glb, result, completed.stdout


def _validate_embedded_textures(glb_path: Path) -> None:
    document = read_glb_json(glb_path)
    images = list(document.get("images") or [])
    if not images:
        raise ArticulatedBakeError("baked GLB contains no images")
    external = [image for image in images if image.get("uri")]
    if external:
        raise ArticulatedBakeError("baked GLB contains external image URIs")
    if any("bufferView" not in image for image in images):
        raise ArticulatedBakeError("baked GLB image is not embedded in a buffer view")


def _tracked_parent_names(document: dict[str, Any]) -> dict[str, str | None]:
    nodes = list(document.get("nodes") or [])
    names = [str(node.get("name") or "") for node in nodes]
    tracked = {
        name
        for name in names
        if name.startswith(LINK_NODE_PREFIX) or name.startswith(VISUAL_NODE_PREFIX)
    }
    parents: dict[str, str | None] = {name: None for name in tracked}
    for parent_index, node in enumerate(nodes):
        parent_name = names[parent_index]
        for child_index in node.get("children") or []:
            child_name = names[int(child_index)]
            if child_name in tracked:
                parents[child_name] = parent_name
    return parents


def _validate_articulation_preserved(
    source_document: dict[str, Any],
    baked_document: dict[str, Any],
) -> None:
    before = _tracked_parent_names(source_document)
    after = _tracked_parent_names(baked_document)
    if set(before) != set(after):
        missing = sorted(set(before) - set(after))
        added = sorted(set(after) - set(before))
        raise ArticulatedBakeError(
            f"Blender changed articulated node names: missing={missing}, added={added}"
        )
    changed = {
        name: {"before": before[name], "after": after[name]}
        for name in sorted(before)
        if before[name] != after[name]
    }
    if changed:
        raise ArticulatedBakeError(f"Blender changed articulated hierarchy: {changed}")


def bake_articulated_glb(
    glb_path: Path,
    renderer_specs: dict[str, dict[str, Any]],
    source_specs: dict[str, dict[str, Any]],
    *,
    policy: BakePolicy | None = None,
) -> dict[str, Any] | None:
    glb_path = Path(glb_path)
    material_specs = collect_texture_bake_specs(renderer_specs, source_specs)
    if not material_specs:
        return None
    resolved_policy = policy or BakePolicy.from_env()
    source_document = read_glb_json(glb_path)
    with tempfile.TemporaryDirectory(prefix=".articulated-bake-", dir=glb_path.parent) as raw:
        temp_dir = Path(raw)
        output, result, _stdout = _run_blender_bake(
            glb_path,
            material_specs,
            temp_dir,
            resolved_policy,
        )
        used_density = int(result.get("texels_per_meter", resolved_policy.texels_per_meter))
        if output.stat().st_size > resolved_policy.soft_size_bytes and (
            used_density > resolved_policy.fallback_texels_per_meter
        ):
            fallback = replace(
                resolved_policy,
                texels_per_meter=resolved_policy.fallback_texels_per_meter,
            )
            output, result, _stdout = _run_blender_bake(
                glb_path,
                material_specs,
                temp_dir,
                fallback,
            )
            result["used_fallback_density"] = True
        size_bytes = output.stat().st_size
        used_density = int(result.get("texels_per_meter", resolved_policy.texels_per_meter))
        if size_bytes > resolved_policy.hard_size_bytes and (
            used_density > resolved_policy.minimum_texels_per_meter
        ):
            minimum = replace(
                resolved_policy,
                texels_per_meter=resolved_policy.minimum_texels_per_meter,
                fallback_texels_per_meter=resolved_policy.minimum_texels_per_meter,
            )
            output, result, _stdout = _run_blender_bake(
                glb_path,
                material_specs,
                temp_dir,
                minimum,
            )
            result["used_minimum_density"] = True
            size_bytes = output.stat().st_size
        if size_bytes > resolved_policy.hard_size_bytes:
            raise ArticulatedBakeError(
                "baked articulated GLB exceeds hard size limit: "
                f"{size_bytes} > {resolved_policy.hard_size_bytes}"
            )
        _validate_embedded_textures(output)
        _validate_articulation_preserved(source_document, read_glb_json(output))
        output.replace(glb_path)
    result["glb_size_bytes"] = glb_path.stat().st_size
    result["source_textures"] = {
        material_name: {
            channel: _sha256_file(Path(path))
            for channel, path in sorted(dict(material_spec["maps"]).items())
        }
        for material_name, material_spec in sorted(material_specs.items())
    }
    return result


__all__ = [
    "BAKE_POLICY_VERSION",
    "ArticulatedBakeError",
    "BakePolicy",
    "bake_articulated_glb",
    "bake_dependency_identity",
    "collect_texture_bake_specs",
]
