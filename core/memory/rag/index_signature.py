"""Compatibility diagnostics for an existing embedding index signature."""

from __future__ import annotations

from typing import Any

from core.i18n import t


def index_signature_error(metadata: Any, model: str, prefix_enabled: bool) -> str | None:
    """Do not infer an old embedding setting from absent or malformed fields.

    Legacy index_meta.json may contain only file hashes. It is not evidence
    that prefixes were disabled. Unknown provenance still requires operator
    verification or a backed-up rebuild; it must not authorize mixed vectors.
    """
    if not isinstance(metadata, dict):
        return t("rag.signature_unknown_shape")
    old_model = metadata.get("embedding_model")
    old_prefix = metadata.get("embedding_e5_prefix")
    if not isinstance(old_model, str) or not old_model or not isinstance(old_prefix, bool):
        return t("rag.signature_unknown_fields")
    if old_model != model:
        return t("rag.signature_model_changed", previous=old_model, current=model)
    if old_prefix != prefix_enabled:
        return t("rag.signature_prefix_changed", previous=old_prefix, current=prefix_enabled)
    return None
