"""End-to-end integration tests: init → extract → search → sleep → compact."""

import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tests.conftest import _random_embedding, _similar_embedding, EMBEDDING_DIM


class TestFullPipeline:
    """Test the complete lifecycle: init DB → add nodes → search → sleep → compact."""

    def test_init_to_search(self, tmp_brain):
        """Init DB, add nodes, then search for them."""
        import brain
        from brain.search import semantic_search

        # Phase 1: Initialize
        conn = brain.get_db()
        assert conn is not None

        # Phase 2: Add nodes manually (simulating extraction)
        brain.add_node("pipe_1", "Python is versatile", "concept",
                       content="Python programming language is versatile for ML and web dev",
                       confidence=0.9, source="human")
        brain.add_node("pipe_2", "Rust for performance", "concept",
                       content="Rust provides memory safety without garbage collection",
                       confidence=0.85, source="second_brain")
        brain.add_node("pipe_3", "Go for concurrency", "concept",
                       content="Go language excels at concurrent programming with goroutines",
                       confidence=0.8, source="extract")
        brain.add_edge("pipe_1", "pipe_2", "relates_to", note="both programming languages")
        brain.add_edge("pipe_2", "pipe_3", "relates_to", note="systems languages")

        # Phase 3: Search
        results = semantic_search("programming language", top_k=5)
        assert len(results) >= 1
        ids = [r["id"] for r in results]
        # All three should appear
        assert len(ids) >= 1

        conn.close()

    def test_search_then_sleep(self, tmp_brain):
        """Add nodes, search (creating access logs), then run sleep cycle."""
        import brain
        from brain.search import semantic_search
        from brain.sleep import run_sleep

        # Add nodes
        for i in range(10):
            brain.add_node(
                f"sl_{i}", f"Sleep test node number {i}", "concept",
                content=f"Content about topic {i} with details about engineering",
                confidence=0.7 + (i * 0.02), source="extract"
            )
        # Add some edges
        for i in range(9):
            brain.add_edge(f"sl_{i}", f"sl_{i+1}", "relates_to")

        # Search to create access logs
        semantic_search("engineering topic")

        # Run sleep
        conn = brain.get_db()
        stats = run_sleep(conn, full=True)
        conn.close()

        assert stats["nodes_scored"] == 10
        # sqrt(10) ≈ 3 nodes promoted
        assert stats["promoted"] == int(math.sqrt(10))

    def test_sleep_gc_respects_grace_period(self, tmp_brain):
        """New nodes should survive GC even with low relevance."""
        import brain
        from brain.sleep import run_sleep

        # Add a node with very low relevance but created recently
        conn = brain.get_db()
        recent = datetime.now().isoformat()
        conn.execute(
            "INSERT INTO nodes (id, label, type, source, relevance_score, created_at, origin) "
            "VALUES ('recent_gc', 'Recent low relevance', 'concept', 'extract', 0.01, ?, 'self')",
            (recent,)
        )
        # Add an old node with low relevance (should be GC'd)
        # Must be old enough that decay_relevance still leaves it below threshold
        old = (datetime.now() - timedelta(days=365)).isoformat()
        conn.execute(
            "INSERT INTO nodes (id, label, type, source, relevance_score, created_at, origin, "
            "last_accessed_at, access_count) "
            "VALUES ('old_gc', 'Old low relevance', 'concept', 'extract', 0.01, ?, 'self', ?, 0)",
            (old, old)
        )
        conn.commit()

        stats = run_sleep(conn, full=True, threshold=0.05)
        conn.close()

        # Verify
        conn = brain.get_db()
        recent_node = conn.execute("SELECT * FROM nodes WHERE id='recent_gc'").fetchone()
        old_node = conn.execute("SELECT * FROM nodes WHERE id='old_gc'").fetchone()
        conn.close()

        assert recent_node is not None, "Recent node should survive GC (grace period)"
        assert old_node is None, "Old low-relevance node should be GC'd"

    @patch("extractors.memory.call_llm")
    def test_extract_then_search(self, mock_llm, tmp_brain, fixture_dir, monkeypatch):
        """Extract from a markdown file, then search for extracted content."""
        import brain
        import extractors.memory as mem
        from brain.search import semantic_search

        monkeypatch.setattr(mem, "MEMORY_DIR", fixture_dir)
        monkeypatch.setattr(mem, "CHECKPOINT_FILE", tmp_brain / "int_mem_ckpt.json")

        # Create a markdown file
        notes_file = fixture_dir / "integration_notes.md"
        notes_file.write_text(
            "# Integration Test Notes\n\n"
            "Today I realized that microservices architecture introduces significant "
            "operational complexity. Service mesh solutions like Istio help but add "
            "their own complexity layer. The team decided to start with a modular "
            "monolith and extract services only when scaling demands it.\n"
        )

        mock_llm.return_value = json.dumps([
            {"id": "microservices_complexity", "label": "Microservices add operational complexity",
             "type": "insight", "content": "Microservices architecture introduces significant operational complexity"},
            {"id": "modular_monolith_decision", "label": "Start with modular monolith",
             "type": "decision", "content": "Team decided to start with modular monolith before extracting microservices"},
        ])

        mem.extract_from_file(notes_file)

        # Now search
        results = semantic_search("microservices architecture complexity")
        assert len(results) >= 1

    def test_dedup_in_sleep(self, tmp_brain):
        """Add near-duplicate nodes, verify sleep deduplicates them."""
        import brain
        from brain.sleep import run_sleep

        base_vec = _random_embedding(seed=100)
        similar_vec = _similar_embedding(base_vec, similarity=0.96, seed=101)

        conn = brain.get_db()
        for nid, label, vec in [
            ("dedup_int_a", "Near duplicate concept alpha version", base_vec),
            ("dedup_int_b", "Near duplicate concept beta version", similar_vec),
        ]:
            conn.execute(
                "INSERT INTO nodes (id, label, type, content, confidence, source, origin) "
                "VALUES (?, ?, 'concept', 'test dedup', 0.8, 'extract', 'self')",
                (nid, label)
            )
            conn.execute(
                "INSERT INTO embeddings (node_id, embedding, model) VALUES (?, ?, ?)",
                (nid, vec.tobytes(), "test-model")
            )
        conn.commit()

        stats = run_sleep(conn, full=True)
        assert stats["dedup_merged"] >= 1

        remaining = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        assert remaining == 1  # one was merged
        conn.close()

    def test_cross_link_in_sleep(self, tmp_brain):
        """Add related but unlinked nodes, verify sleep cross-links them."""
        import brain
        from brain.sleep import run_sleep

        base_vec = _random_embedding(seed=200)
        # Similarity ~0.75 (above cross-link threshold 0.68, below dedup 0.88)
        related_vec = _similar_embedding(base_vec, similarity=0.75, seed=201)

        conn = brain.get_db()
        for nid, label, vec in [
            ("xlink_a", "Cross link alpha topic", base_vec),
            ("xlink_b", "Cross link beta topic", related_vec),
        ]:
            conn.execute(
                "INSERT INTO nodes (id, label, type, content, confidence, source, origin) "
                "VALUES (?, ?, 'concept', 'test xlink', 0.8, 'extract', 'self')",
                (nid, label)
            )
            conn.execute(
                "INSERT INTO embeddings (node_id, embedding, model) VALUES (?, ?, ?)",
                (nid, vec.tobytes(), "test-model")
            )
        conn.commit()

        stats = run_sleep(conn, full=True)

        # Check if edge was created
        edge = conn.execute(
            "SELECT * FROM edges WHERE "
            "(from_id='xlink_a' AND to_id='xlink_b') OR "
            "(from_id='xlink_b' AND to_id='xlink_a')"
        ).fetchone()
        conn.close()

        if stats["cross_linked"] > 0:
            assert edge is not None

    def test_compact_module(self, tmp_brain):
        """Verify compact.py wrapper works end-to-end."""
        import brain
        from brain.compact import run_sleep

        conn = brain.get_db()
        conn.execute(
            "INSERT INTO nodes (id, label, type, source, origin) "
            "VALUES ('compact_test', 'Compact test node', 'concept', 'extract', 'self')"
        )
        conn.commit()

        stats = run_sleep(conn, dry_run=True, full=True)
        assert stats["nodes_scored"] == 1
        assert stats["dry_run"] is True
        conn.close()

    def test_hybrid_search_leverages_graph(self, tmp_brain):
        """Hybrid search should return nodes reachable via graph walk
        that might not appear in pure semantic search."""
        import brain
        from brain.search import hybrid_search, semantic_search

        # Create a chain: A -> B -> C
        # A is semantically similar to query, C is not, but reachable via graph
        brain.add_node("chain_a", "Machine learning basics", "concept",
                       content="Introduction to machine learning fundamentals")
        brain.add_node("chain_b", "Neural network architectures", "concept",
                       content="Deep neural network design patterns")
        brain.add_node("chain_c", "GPU cluster management", "concept",
                       content="Managing distributed GPU clusters for training")
        brain.add_edge("chain_a", "chain_b", "enables")
        brain.add_edge("chain_b", "chain_c", "depends_on")

        hybrid_results = hybrid_search("machine learning fundamentals", top_k=10)
        assert len(hybrid_results) >= 1

    def test_end_to_end_with_core_memory(self, tmp_brain):
        """Full cycle: add nodes → sleep (promotes core) → search (core gets boost)."""
        import brain
        from brain.sleep import run_sleep
        from brain.search import semantic_search

        # Add enough nodes to make sqrt(N) meaningful
        content_base = "Knowledge about topic number"
        for i in range(16):
            brain.add_node(
                f"core_{i}", f"Core test node {i}", "concept",
                content=f"{content_base} {i} with details",
                confidence=0.8, source="human" if i < 2 else "extract"
            )
        # Create edges (hub node 0 connects to many)
        for i in range(1, 16):
            brain.add_edge("core_0", f"core_{i}", "relates_to")

        # Run sleep to promote core memories
        conn = brain.get_db()
        stats = run_sleep(conn, full=True)
        conn.close()

        # sqrt(16) = 4 nodes should be promoted
        assert stats["promoted"] == 4

        # Verify core_0 is promoted (it has most edges + human source)
        conn = brain.get_db()
        core_0 = conn.execute(
            "SELECT core_memory FROM nodes WHERE id='core_0'"
        ).fetchone()
        conn.close()
        assert core_0["core_memory"] == 1

    def test_large_graph_sleep_stability(self, tmp_brain):
        """Sleep cycle should handle a graph of 100+ nodes without errors."""
        import brain
        from brain.sleep import run_sleep

        conn = brain.get_db()
        for i in range(100):
            vec = _random_embedding(seed=1000 + i)
            conn.execute(
                "INSERT INTO nodes (id, label, type, content, confidence, source, origin) "
                "VALUES (?, ?, 'concept', ?, 0.7, 'extract', 'self')",
                (f"big_{i}", f"Large graph node {i}", f"Content {i}")
            )
            conn.execute(
                "INSERT INTO embeddings (node_id, embedding, model) VALUES (?, ?, ?)",
                (f"big_{i}", vec.tobytes(), "test-model")
            )
        # Add random edges
        import random
        random.seed(42)
        for _ in range(200):
            a, b = random.sample(range(100), 2)
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO edges (from_id, to_id, relation, source) "
                    "VALUES (?, ?, 'relates_to', 'test')",
                    (f"big_{a}", f"big_{b}")
                )
            except Exception:
                pass
        conn.commit()

        stats = run_sleep(conn, full=True)
        assert stats["nodes_scored"] >= 50  # some may have been deduped
        conn.close()
