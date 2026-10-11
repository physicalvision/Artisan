from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


def main() -> int:
    try:
        request = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError as exc:
        sys.stdout.write(
            json.dumps({"ok": False, "error": {"type": "bad_request", "message": str(exc)}})
        )
        return 0

    try:
        from sdk.runtime.runtime import probe_summary

        manifest = None
        if request.get("inspection_snapshot") is not None:
            from sdk.runtime.inspection import _load_reference

            manifest = _load_reference(
                request["inspection_snapshot"],
                Path(str(request["file_path"])),
                str(request.get("sdk_package") or "sdk"),
            )

        from sdk.runtime.accepted_inspection import accepted_inspection

        with accepted_inspection(
            Path(str(request["file_path"])), request.get("accepted_candidate")
        ):
            result = probe_summary(
                Path(str(request.get("file_path"))),
                sdk_package=str(request.get("sdk_package") or "sdk"),
                **({"collision_manifest": manifest} if manifest is not None else {}),
            )
        payload = {"ok": True, "result": result}
    except Exception as exc:  # noqa: BLE001 — any failure must come back as a payload
        payload = {
            "ok": False,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc()[-4000:],
            },
        }

    sys.stdout.write(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
