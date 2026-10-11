from __future__ import annotations

from pathlib import Path

from pipeline.backends.base import CompileOutcome
from sdk._resources import SDK_ROOT
from sdk.runtime import runtime as sdk_runtime

SDK_CORPUS_ROOT = SDK_ROOT
CONTRACT_DOC = SDK_ROOT / "_docs" / "model_contract.md"


class SdkBackend:
    name = "sdk_build"

    def __init__(self, *, sdk_package: str = "sdk") -> None:
        self.sdk_package = sdk_package

    def compile_gate(self, artifact_path: Path) -> CompileOutcome:
        report = artifact_path.parent / "_compile_report.json"
        return sdk_runtime.compile_gate(artifact_path, report, sdk_package=self.sdk_package)

    def render_views(
        self,
        artifact_path: Path,
        views: list[str],
        out_dir: Path,
        *,
        pose: dict[str, float] | None = None,
        camera: dict[str, float] | None = None,
        focus: list[str] | None = None,
        segment: bool = False,
        inspection_snapshot: dict[str, str] | None = None,
    ) -> list[Path]:
        from sdk.runtime import render as sdk_render

        return sdk_render.render_views(
            artifact_path,
            views,
            out_dir,
            sdk_package=self.sdk_package,
            pose=pose,
            camera=camera,
            focus=focus,
            segment=segment,
            **({"inspection_snapshot": inspection_snapshot} if inspection_snapshot else {}),
        )

    def export(self, artifact_path: Path, formats: list[str], out_dir: Path):
        from sdk.runtime import render as sdk_render

        return sdk_render.export(artifact_path, formats, out_dir, sdk_package=self.sdk_package)

    def publish_release_candidate(
        self,
        manifest_path: Path,
        out_dir: Path,
        *,
        expected_candidate_id: str | None = None,
    ):
        from sdk.runtime.release_candidate import publish_release_candidate

        return publish_release_candidate(
            manifest_path,
            out_dir,
            expected_candidate_id=expected_candidate_id,
        )

    async def code_step(self, state, ctx, *, instruction: str, session_path):
        raise NotImplementedError("driven via pipeline.agents.coding.CodingAgent")
