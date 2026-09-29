"""Merge human-written meaning into discovered manifests (e.g. 'dx_code is ICD-10-CM')."""

from __future__ import annotations

from typing import Any

from agentic_search.core.types import FieldSpec, FieldType, Manifest

_FIELD_KEYS = ("description", "embedder_id", "vector_dim", "vector_metric",
               "searchable", "filterable", "sortable")


def apply_annotations(manifest: Manifest, ann: dict[str, Any] | None) -> Manifest:
    """Return a copy of `manifest` with annotations applied. Unknown collections are ignored;
    unknown fields are added (useful to declare vector fields introspection can't see)."""
    if not ann:
        return manifest
    out = manifest.model_copy(deep=True)
    if "description" in ann:
        out.description = ann["description"]
    for cname, cann in (ann.get("collections") or {}).items():
        coll = out.resolve_collection(cname)
        if coll is None:
            continue
        if "description" in cann:
            coll.description = cann["description"]
        for fname, fann in (cann.get("fields") or {}).items():
            spec = coll.field(fname)
            if spec is None:
                spec = FieldSpec(name=fname, type=FieldType(fann.get("type", "keyword")))
                coll.fields.append(spec)
            elif "type" in fann:
                spec.type = FieldType(fann["type"])
            for key in _FIELD_KEYS:
                if key in fann:
                    setattr(spec, key, fann[key])
    return out
