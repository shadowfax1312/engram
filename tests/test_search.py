"""Search pipeline tests."""
import sys, os
sys.path.insert(0, os.path.expanduser("~/.openclaw/workspace/graph"))
import pytest
from search import semantic_search, hybrid_search


def test_semantic_search_returns_results():
    results = semantic_search("consciousness", top_k=5)
    assert len(results) > 0
    assert len(results) <= 5


def test_semantic_search_result_keys():
    results = semantic_search("decision making", top_k=3)
    assert len(results) > 0
    for r in results:
        for key in ("id", "label", "content", "score"):
            assert key in r, f"Missing key {key} in result"


def test_semantic_search_origin_onyx():
    results = semantic_search("self awareness", top_k=5, origin="onyx")
    for r in results:
        assert r.get("origin") == "onyx", f"Expected origin=onyx, got {r.get('origin')}"


def test_semantic_search_origin_self():
    results = semantic_search("portfolio investing", top_k=5, origin="self")
    for r in results:
        assert r.get("origin") == "self", f"Expected origin=self, got {r.get('origin')}"


def test_semantic_search_exclude_external():
    results = semantic_search("machine learning", top_k=10, include_external=False)
    for r in results:
        assert r.get("origin") in (None, "self", "onyx"), f"External node leaked: {r.get('origin')}"


def test_hybrid_search_returns_results():
    results = hybrid_search("consciousness", top_k=5)
    assert len(results) > 0


def test_hybrid_search_origin_filter():
    results = hybrid_search("reflection", top_k=5, origin="onyx")
    for r in results:
        assert r.get("origin") == "onyx", f"Expected origin=onyx, got {r.get('origin')}"


def test_hybrid_search_score_present():
    results = hybrid_search("decision", top_k=3)
    for r in results:
        assert "score" in r
        assert r["score"] >= 0
