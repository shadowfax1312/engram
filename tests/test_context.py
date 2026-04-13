"""Context pipeline tests."""
import sys, os
sys.path.insert(0, os.path.expanduser("~/.openclaw/workspace/graph"))
import pytest
from context import get_context, get_context_string


def test_get_context_string_nonempty():
    result = get_context_string("consciousness", top_k=3)
    assert isinstance(result, str)
    assert len(result) > 10


def test_get_context_no_relation_in_parent_chain():
    nodes = get_context("decision making", top_k=5)
    for node in nodes:
        for p in node.parent_chain:
            assert "relation" not in p, f"relation key should be gone from parent_chain, got: {p}"


def test_get_context_origin_scoped():
    """origin filter scopes results — verify via search layer (RelevantNode has no origin field)."""
    import sys, os
    sys.path.insert(0, os.path.expanduser("~/.openclaw/workspace/graph"))
    from search import hybrid_search
    results = hybrid_search("self awareness", top_k=5, origin="onyx")
    for r in results:
        assert r.get("origin") == "onyx", f"Expected origin=onyx, got {r.get('origin')}"


def test_get_context_string_origin():
    result = get_context_string("reflection", top_k=3, origin="onyx")
    assert isinstance(result, str)
    assert len(result) > 0
