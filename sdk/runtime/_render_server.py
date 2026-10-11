from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _render_driver as drv  # type: ignore  # noqa: E402
from _render_protocol import input_identity, verify_input, write_report  # noqa: E402


def _handle(req: dict) -> str:
    report_path = req["report"]
    out = {"glb": req.get("glb"), "renders": {}, "errors": []}
    try:
        expected = req.get("input_identity") or input_identity(Path(req["glb"]))
        out.update(request_id=req.get("request_id"), input_identity=expected)
        verify_input(Path(req["glb"]), expected)
        drv._clear_imported()
        started = time.monotonic()
        drv._import_glb(req["glb"])
        out["import_wall_s"] = time.monotonic() - started
        views = [v for v in (req.get("views") or []) if v]
        out["renders"] = drv._render_views_only(views, Path(req["out_dir"]), req.get("camera", ""))
        verify_input(Path(req["glb"]), expected)
    except Exception as exc:  # noqa: BLE001 — report the error, keep the server alive
        out["error_kind"] = type(exc).__name__
        out["errors"].append(traceback.format_exc())
    out["device"] = drv._DEVICE_INFO
    out["view_timings"] = drv._VIEW_TIMINGS
    write_report(Path(report_path), out)
    return report_path


def main() -> None:
    drv._reset_scene()
    drv._configure_scene()
    print("__SERVER_READY__", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        if req.get("cmd") == "quit":
            break
        rep = _handle(req)
        print(f"__RENDER_DONE__ {req.get('request_id') or rep}", flush=True)


if __name__ == "__main__":
    main()
