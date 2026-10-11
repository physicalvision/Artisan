from .cadquery import (
    CadQueryMeshExport,
    cadquery_anchor,
    cadquery_local_aabb,
    export_cadquery_components,
    export_cadquery_mesh,
    mesh_components_from_cadquery,
    mesh_from_cadquery,
    tessellate_cadquery,
)

__all__ = [
    "CadQueryMeshExport",
    "cadquery_anchor",
    "cadquery_local_aabb",
    "export_cadquery_components",
    "export_cadquery_mesh",
    "mesh_components_from_cadquery",
    "mesh_from_cadquery",
    "tessellate_cadquery",
]


def __getattr__(name: str):
    if name == "save_cadquery_obj":
        from .cadquery import save_cadquery_obj

        globals()[name] = save_cadquery_obj
        return save_cadquery_obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
