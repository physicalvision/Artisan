from __future__ import annotations

import argparse
import inspect
import json
import sys

import bmesh  # type: ignore[import-not-found]
import bpy  # type: ignore[import-not-found]

OP_CATEGORIES = ("mesh", "object", "transform", "curve")


def _parse():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def _op_entry(cat: str, name: str):
    op = getattr(getattr(bpy.ops, cat), name)
    rna = op.get_rna_type()
    args, doc_args = [], []
    for prop in rna.properties:
        if prop.identifier == "rna_type":
            continue
        args.append(prop.identifier)
        desc = (prop.description or "").strip()
        doc_args.append(f"{prop.identifier}: {desc}" if desc else prop.identifier)
    summary = (rna.description or "").strip()
    return {
        "qual": f"bpy.ops.{cat}.{name}",
        "name": name,
        "signature": "(" + ", ".join(args) + ")",
        "summary": summary,
        "doc": summary + (" | args: " + "; ".join(doc_args) if doc_args else ""),
        "kind": "op",
    }


def _bmesh_entry(name: str):
    doc = inspect.getdoc(getattr(bmesh.ops, name)) or ""
    first = next((line.strip() for line in doc.splitlines() if line.strip()), "")
    return {
        "qual": f"bmesh.ops.{name}",
        "name": name,
        "signature": "",
        "summary": first,
        "doc": doc,
        "kind": "bmesh_op",
    }


def _modifier_entry(tname: str):
    brna = getattr(bpy.types, tname).bl_rna
    props = []
    for p in brna.properties:
        if p.identifier == "rna_type":
            continue
        desc = (p.description or "").strip()
        props.append(f"{p.identifier}: {desc}" if desc else p.identifier)
    summary = (brna.description or "").strip()
    return {
        "qual": f"bpy.types.{tname}",
        "name": tname,
        "signature": "",
        "summary": summary,
        "doc": summary + (" | props: " + "; ".join(props[:48]) if props else ""),
        "kind": "modifier",
    }


def main():
    entries = []
    for cat in OP_CATEGORIES:
        catmod = getattr(bpy.ops, cat, None)
        if catmod is None:
            continue
        for name in dir(catmod):
            if name.startswith("_"):
                continue
            try:
                entries.append(_op_entry(cat, name))
            except Exception:  # noqa: BLE001 — skip operators without a usable RNA
                pass
    for name in dir(bmesh.ops):
        if not name.startswith("_"):
            entries.append(_bmesh_entry(name))
    for tname in dir(bpy.types):
        if tname.endswith("Modifier"):
            try:
                entries.append(_modifier_entry(tname))
            except Exception:  # noqa: BLE001
                pass

    out = {"version": bpy.app.version_string, "entries": entries}
    args = _parse()
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"__WROTE__ {len(entries)} entries -> {args.out}")


if __name__ == "__main__":
    main()
