from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import lru_cache

from pipeline import retrieval

SYN: dict[str, list[str]] = {
    "hollow": ["shell"],
    "hollowed": ["shell"],
    "empty": ["shell"],
    "thin": ["shell"],
    "wall": ["shell"],
    "shelled": ["shell"],
    "casing": ["shell"],
    "enclosure": ["shell"],
    "housing": ["shell"],
    "round": ["fillet"],
    "rounded": ["fillet"],
    "radius": ["fillet"],
    "bevel": ["chamfer"],
    "beveled": ["chamfer"],
    "chamfered": ["chamfer"],
    "hole": ["hole", "cborehole", "cskhole"],
    "bore": ["cborehole", "hole"],
    "drill": ["hole"],
    "counterbore": ["cborehole"],
    "countersink": ["cskhole"],
    "bolt": ["cborehole", "cskhole", "hole"],
    "screw": ["cborehole", "hole"],
    "pattern": ["rarray", "polararray"],
    "array": ["rarray", "polararray"],
    "grid": ["rarray"],
    "circular": ["polararray"],
    "polar": ["polararray"],
    "repeat": ["rarray", "polararray"],
    "duplicate": ["rarray"],
    "taper": ["loft", "revolve"],
    "tapered": ["loft", "revolve"],
    "frustum": ["loft", "revolve"],
    "cone": ["cone", "loft", "revolve"],
    "conical": ["loft", "revolve"],
    "lofted": ["loft"],
    "extruded": ["extrude"],
    "swept": ["sweep"],
    "revolved": ["revolve"],
    "pipe": ["sweep"],
    "mirror": ["mirror"],
    "reflect": ["mirror"],
    "flip": ["mirror"],
    "move": ["translate"],
    "shift": ["translate"],
    "rotate": ["rotate"],
    "max": ["maximize", "selector"],
    "maximum": ["maximize", "selector"],
    "highest": ["maximize", "selector"],
    "top": ["maximize", "selector"],
    "upper": ["maximize", "selector"],
    "topmost": ["maximize", "selector"],
    "min": ["minimize", "selector"],
    "minimum": ["minimize", "selector"],
    "lowest": ["minimize", "selector"],
    "bottom": ["minimize", "selector"],
    "lower": ["minimize", "selector"],
    "parallel": ["parallel", "selector"],
    "perpendicular": ["perpendicular", "selector"],
    "select": ["faces", "edges", "vertices", "selector"],
    "pick": ["faces", "edges", "vertices", "selector"],
    "boss": ["extrude"],
    "rib": ["extrude"],
    "pocket": ["cut"],
    "groove": ["cut"],
    "slot": ["slot", "cut"],
    "text": ["text"],
    "thread": ["thread"],
    "split": ["split"],
    "section": ["section"],
}

CLS_WEIGHT: dict[str, float] = {
    "Workplane": 1.35,
    "Sketch": 1.15,
    "cq": 1.1,
    "selectors": 1.2,
    "Assembly": 1.0,
    "Solid": 0.7,
    "Face": 0.6,
    "Edge": 0.6,
    "Wire": 0.6,
    "Shape": 0.6,
    "Vertex": 0.55,
    "Vector": 0.55,
    "Plane": 0.6,
}
CLS_RANK: dict[str, int] = {
    "Workplane": 0,
    "Sketch": 1,
    "cq": 2,
    "selectors": 3,
    "Assembly": 4,
    "Solid": 5,
    "Shape": 6,
    "Face": 7,
    "Edge": 8,
    "Wire": 9,
    "Vertex": 10,
    "Vector": 11,
    "Plane": 12,
}

INDEX_CLASS_NAMES = (
    "Workplane",
    "Sketch",
    "Assembly",
    "Shape",
    "Solid",
    "Face",
    "Edge",
    "Wire",
    "Vertex",
    "Vector",
    "Plane",
)


@dataclass(slots=True, frozen=True)
class ApiEntry:
    qual: str
    name: str
    signature: str
    summary: str
    doc: str
    kind: str


def _first_line(doc: str) -> str:
    for line in doc.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _make_entry(qual: str, name: str, obj: object, kind: str) -> ApiEntry:
    doc = inspect.getdoc(obj) or ""
    try:
        sig = str(inspect.signature(obj))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        sig = ""
    return ApiEntry(
        qual=qual,
        name=name,
        signature=sig,
        summary=_first_line(doc),
        doc=doc,
        kind=kind,
    )


def build_index() -> list[ApiEntry]:
    import cadquery as cq
    import cadquery.selectors as cqsel

    index: list[ApiEntry] = []
    for cname in INDEX_CLASS_NAMES:
        cls = getattr(cq, cname, None)
        if cls is None:
            continue
        for member in dir(cls):
            if member.startswith("_"):
                continue
            obj = getattr(cls, member, None)
            if callable(obj):
                kind = "accessor" if member[:1].isupper() else "method"
                index.append(_make_entry(f"{cname}.{member}", member, obj, kind))
    for sname in dir(cqsel):
        obj = getattr(cqsel, sname, None)
        if inspect.isclass(obj) and sname.endswith("Selector"):
            index.append(_make_entry(f"selectors.{sname}", sname, obj, "selector"))
    for fname in dir(cq):
        obj = getattr(cq, fname, None)
        if inspect.isfunction(obj):
            index.append(_make_entry(f"cq.{fname}", fname, obj, "free_fn"))
    return index


def _build_doc_tokens(entry: ApiEntry) -> list[str]:
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
class _CqLookup:
    entries: list[ApiEntry]
    bm25: retrieval.BM25Index


@lru_cache(maxsize=1)
def get_cq_lookup() -> _CqLookup:
    entries = build_index()
    doc_token_lists = [_build_doc_tokens(e) for e in entries]
    bm25 = retrieval.build_bm25(doc_token_lists)
    return _CqLookup(entries=entries, bm25=bm25)


def lookup_api(query: str, *, k: int = 5, full: bool = False, klass: str | None = None) -> str:
    lk = get_cq_lookup()
    qtok = expand_query(query)
    if not qtok:
        return "No matching API."

    raw_hits = retrieval.search_bm25(lk.bm25, qtok, k=30)
    cand: list[tuple[float, ApiEntry]] = []
    for idx, raw in raw_hits:
        entry = lk.entries[idx]
        cls = entry.qual.split(".", 1)[0]
        score = raw * CLS_WEIGHT.get(cls, 0.8)
        if entry.kind == "accessor":
            score *= 0.35
        cand.append((score, entry))

    if klass:
        prefix = f"{klass.lower()}."
        cand = [(s, e) for s, e in cand if e.qual.lower().startswith(prefix)]

    cand.sort(
        key=lambda x: (
            -x[0],
            CLS_RANK.get(x[1].qual.split(".", 1)[0], 99),
            len(x[1].qual),
        )
    )

    seen: set[str] = set()
    out: list[ApiEntry] = []
    for _score, entry in cand:
        if entry.name in seen:
            continue
        seen.add(entry.name)
        out.append(entry)
        if len(out) >= k:
            break

    if not out:
        return "No matching API."

    if full:
        e = out[0]
        return f"{e.qual}{e.signature}\n\n{e.doc or '(no docstring)'}"

    lines = ["cadquery API (introspected from the installed version) — top matches:"]
    for e in out:
        sig = e.signature if len(e.signature) <= 70 else e.signature[:69] + "…"
        summary = e.summary if len(e.summary) <= 100 else e.summary[:99] + "…"
        lines.append(f"{e.qual}{sig} — {summary}")
    lines.append("Pass full=true with a focused query to read one entry's complete docstring.")
    return "\n".join(lines)
