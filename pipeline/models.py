from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(slots=True, frozen=True)
class CompileSignal:
    severity: Literal["failure", "warning", "note"]
    kind: str
    code: str
    summary: str
    details: str = ""
    blocking: bool = False
    source: Literal["compiler", "tests", "harness"] = "compiler"
    group: Literal["build", "qc", "design", "hygiene"] = "qc"
    check_name: str | None = None
    dedupe_key: str | None = None
    evidence: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "kind": self.kind,
            "code": self.code,
            "summary": self.summary,
            "details": self.details,
            "blocking": self.blocking,
            "source": self.source,
            "group": self.group,
            "check_name": self.check_name,
            "dedupe_key": self.dedupe_key,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CompileSignal":
        return cls(
            severity=str(payload.get("severity", "warning")),  # type: ignore[arg-type]
            kind=str(payload.get("kind", "unknown")),
            code=str(payload.get("code", "UNKNOWN")),
            summary=str(payload.get("summary", "")),
            details=str(payload.get("details", "")),
            blocking=bool(payload.get("blocking", False)),
            source=str(payload.get("source", "compiler")),  # type: ignore[arg-type]
            group=str(payload.get("group", "qc")),  # type: ignore[arg-type]
            check_name=(
                None if payload.get("check_name") is None else str(payload.get("check_name"))
            ),
            dedupe_key=(
                None if payload.get("dedupe_key") is None else str(payload.get("dedupe_key"))
            ),
            evidence=(
                dict(payload["evidence"]) if isinstance(payload.get("evidence"), dict) else None
            ),
        )


@dataclass(slots=True, frozen=True)
class CompileSignalBundle:
    status: Literal["success", "failure"]
    summary: str
    signals: tuple[CompileSignal, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "summary": self.summary,
            "signals": [signal.to_dict() for signal in self.signals],
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CompileSignalBundle":
        raw_signals = payload.get("signals")
        signals: list[CompileSignal] = []
        if isinstance(raw_signals, list):
            for item in raw_signals:
                if isinstance(item, dict):
                    signals.append(CompileSignal.from_dict(item))
        return cls(
            status=str(payload.get("status", "success")),  # type: ignore[arg-type]
            summary=str(payload.get("summary", "")),
            signals=tuple(signals),
            evidence=(
                dict(payload["evidence"]) if isinstance(payload.get("evidence"), dict) else {}
            ),
        )


@dataclass(slots=True, frozen=True)
class CompileReport:

    urdf_xml: str
    warnings: list[str]
    signal_bundle: CompileSignalBundle
    diagnostics: dict[str, Any] | None = None
