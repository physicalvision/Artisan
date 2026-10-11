from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pipeline import retrieval

_INTROSPECT = Path(__file__).resolve().parent / "blender" / "_introspect_api.py"

SYN: dict[str, list[str]] = {
    "smooth": ["subsurf", "subdivision", "shade"],
    "subdivide": ["subdivide", "subsurf", "subdivision"],
    "subdivision": ["subsurf", "subdivision"],
    "round": ["bevel", "subsurf"],
    "rounded": ["bevel"],
    "fillet": ["bevel"],
    "bevel": ["bevel"],
    "thicken": ["solidify"],
    "thickness": ["solidify"],
    "shell": ["solidify"],
    "hollow": ["solidify"],
    "displace": ["displace"],
    "displacement": ["displace"],
    "bump": ["displace"],
    "wrinkle": ["displace", "noise"],
    "fold": ["displace", "cloth"],
    "noise": ["displace"],
    "wrap": ["shrinkwrap"],
    "shrink": ["shrinkwrap"],
    "simplify": ["decimate"],
    "reduce": ["decimate"],
    "lowpoly": ["decimate"],
    "decimate": ["decimate", "collapse"],
    "remesh": ["remesh"],
    "voxel": ["remesh"],
    "cloth": ["cloth"],
    "soft": ["cloth", "softbody", "subsurf"],
    "drape": ["cloth"],
    "sag": ["cloth"],
    "extrude": ["extrude"],
    "inset": ["inset"],
    "loopcut": ["loopcut"],
    "bridge": ["bridge"],
    "fill": ["fill", "grid", "triangulate"],
    "merge": ["merge", "weld", "remove_doubles"],
    "weld": ["weld", "merge"],
    "mirror": ["mirror", "symmetrize"],
    "deform": ["lattice", "simple_deform", "cast"],
    "bend": ["simple_deform"],
    "taper": ["simple_deform", "cast"],
    "twist": ["simple_deform"],
    "skin": ["skin"],
    "metaball": ["meta"],
    "normals": ["normals", "recalc", "flip"],
    "recalculate": ["recalc", "normals"],
    "tuft": ["poke", "inset"],
    "button": ["poke"],
    "pucker": ["poke"],
    "seam": ["mark_seam", "crease"],
    "crease": ["crease"],
    "cube": ["primitive_cube"],
    "box": ["primitive_cube"],
    "sphere": ["primitive_uv_sphere", "primitive_ico_sphere"],
    "cylinder": ["primitive_cylinder"],
    "cone": ["primitive_cone"],
    "plane": ["primitive_plane", "grid"],
    "grid": ["primitive_grid"],
    "circle": ["primitive_circle"],
    "torus": ["primitive_torus"],
    "scale": ["resize"],
    "move": ["translate"],
}

KIND_WEIGHT: dict[str, float] = {"modifier": 1.30, "bmesh_op": 1.20, "op": 1.0}
KIND_RANK: dict[str, int] = {"modifier": 0, "bmesh_op": 1, "op": 2}


@dataclass(slots=True, frozen=True)
class ApiEntry:
    qual: str
    name: str
    signature: str
    summary: str
    doc: str
    kind: str


def _index_json_path() -> Path:
    from artisan.config import cache_dir
    from pipeline.blender.runtime import blender_bin

    tag = hashlib.sha256(blender_bin().encode("utf-8")).hexdigest()[:10]
    return cache_dir() / "blender_api" / f"{tag}.json"


def _ensure_index_json() -> Path:
    out = _index_json_path()
    if out.exists():
        return out
    from pipeline.blender.runtime import blender_bin

    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        blender_bin(),
        "--background",
        "--factory-startup",
        "--python",
        str(_INTROSPECT),
        "--",
        "--out",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if not out.exists():
        raise RuntimeError(
            f"blender API introspection produced no index (rc={proc.returncode}).\n"
            f"STDERR:\n{proc.stderr[-1500:]}"
        )
    return out


def build_index() -> list[ApiEntry]:
    data = json.loads(_ensure_index_json().read_text(encoding="utf-8"))
    return [ApiEntry(**e) for e in data.get("entries", [])]


def _doc_tokens(entry: ApiEntry) -> list[str]:
    return retrieval.weighted_doc(
        (retrieval.identifier_tokens(entry.name), 3),
        (retrieval.tokenize(entry.summary), 2),
        (retrieval.tokenize(entry.doc), 1),
    )


def expand_query(query: str) -> list[str]:
    qtok = retrieval.tokenize(query)
    extra: list[str] = []
    for token in qtok:
        extra.extend(SYN.get(token, []))
    return qtok + extra


@dataclass(slots=True)
class _BpyLookup:
    entries: list[ApiEntry]
    bm25: retrieval.BM25Index


@lru_cache(maxsize=1)
def get_blender_lookup() -> _BpyLookup:
    entries = build_index()
    bm25 = retrieval.build_bm25([_doc_tokens(e) for e in entries])
    return _BpyLookup(entries=entries, bm25=bm25)


def lookup_blender_api(query: str, *, k: int = 6, full: bool = False) -> str:
    lk = get_blender_lookup()
    qtok = expand_query(query)
    if not qtok:
        return "No matching API."

    raw_hits = retrieval.search_bm25(lk.bm25, qtok, k=30)
    cand: list[tuple[float, ApiEntry]] = []
    for idx, raw in raw_hits:
        entry = lk.entries[idx]
        cand.append((raw * KIND_WEIGHT.get(entry.kind, 1.0), entry))
    cand.sort(key=lambda x: (-x[0], KIND_RANK.get(x[1].kind, 9), len(x[1].qual)))

    seen: set[str] = set()
    out: list[ApiEntry] = []
    for _score, entry in cand:
        if entry.qual in seen:
            continue
        seen.add(entry.qual)
        out.append(entry)
        if len(out) >= k:
            break
    if not out:
        return "No matching API."

    if full:
        e = out[0]
        return f"{e.qual}{e.signature}\n\n{e.doc or '(no docstring)'}"

    lines = ["Blender bpy API (introspected from the installed Blender) — top matches:"]
    for e in out:
        sig = e.signature if len(e.signature) <= 80 else e.signature[:79] + "…"
        summary = e.summary if len(e.summary) <= 110 else e.summary[:109] + "…"
        lines.append(f"{e.qual}{sig} — {summary}")
    lines.append("Pass full=true with a focused query to read one entry's complete doc + args.")
    return "\n".join(lines)
