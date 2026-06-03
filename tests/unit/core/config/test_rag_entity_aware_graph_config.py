from __future__ import annotations

from core.config.schemas import RAGConfig


def test_entity_aware_graph_defaults_disabled() -> None:
    """Entity-aware Legacy graph is config-controlled and disabled by default."""
    rag = RAGConfig()
    assert rag.entity_aware_graph_enabled is False
    assert rag.graph_entity_edge_cap == 8
    assert rag.graph_inverse_fan_enabled is True
    assert rag.graph_recency_weight_enabled is True
