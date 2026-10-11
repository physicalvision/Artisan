from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pipeline.models import CompileReport
    from pipeline.state import PipelineState


@dataclass(slots=True)
class CompileOutcome:

    status: str
    summary: str
    report_path: str | None = None
    blocking_signals: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw: "CompileReport | None" = None

    def ok(self) -> bool:
        return self.status == "success" and not self.blocking_signals


@dataclass(slots=True)
class CodeResult:
    asset_build_path: str
    compile: CompileOutcome
    cost_usd: float = 0.0
    session_path: str | None = None
    turns: int = 0


@runtime_checkable
class ImplementationBackend(Protocol):
    name: str

    async def code_step(
        self,
        state: "PipelineState",
        ctx: Any,
        *,
        instruction: str,
        session_path: str | None,
    ) -> CodeResult:
        pass

    def compile_gate(self, artifact_path: Path) -> CompileOutcome:
        pass

    def render_views(self, artifact_path: Path, views: list[str], out_dir: Path) -> list[Path]:
        pass

    def export(self, artifact_path: Path, formats: list[str], out_dir: Path) -> Any:
        pass

    def publish_release_candidate(
        self,
        manifest_path: Path,
        out_dir: Path,
        *,
        expected_candidate_id: str | None = None,
    ) -> Any:
        pass
