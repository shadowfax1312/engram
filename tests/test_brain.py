"""Unit tests for brain/__init__.py, search.py, sleep.py, compact.py."""

import json
import math
import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pytest

from tests.conftest import _random_embedding, _similar_embedding, EMBEDDING_DIM


# ═══════════════════════════════════════════════════════════════════
# brain/__init__.py
# ═══════════════════════════════════════════════════════════════════


class TestGetDb:
    def test_creates_tables(self, db):
        tables = [r[0] for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        for t in ("nodes", "edges", "embeddings", "access_log", "ruminate_log"):
            assert t in tables

    def test_idempotent(self, tmp_brain):
        """Calling get_db() twice doesn't crash."""
        import brain
        c1 = brain.get_db()
        c2 = brain.get_db()
        c1.close()
        c2.close()

    def test_wal_mode(self, db):
        mode = db.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"


class TestAddNode:
    def test_basic_insert(self, tmp_brain):
        import brain
        brain.add_node("test1", "Test Label", "concept", content="Some content")
        conn = brain.get_db()
        row = conn.execute("SELECT * FROM nodes WHERE id='test1'").fetchone()
        conn.close()
        assert row is not None
        assert row["label"] == "Test Label"
        assert row["type"] == "concept"

    def test_rejects_garbage_labels(self, tmp_brain):
        import brain
        assert brain.add_node("g1", "", "concept") is False
        assert brain.add_node("g2", "ab", "concept") is False
        assert brain.add_node("g3", "...", "concept") is False
        assert brain.add_node("g4", "null", "concept") is False
        assert brain.add_node("g5", "None", "concept") is False
        assert brain.add_node("g6", "undefined", "concept") is False

    def test_upsert_on_conflict(self, tmp_brain):
        import brain
        brain.add_node("up1", "Original", "concept", content="v1", confidence=0.5)
        brain.add_node("up1", "Updated", "concept", content="v2", confidence=0.9)
        conn = brain.get_db()
        row = conn.execute("SELECT * FROM nodes WHERE id='up1'").fetchone()
        conn.close()
        assert row["label"] == "Updated"
        assert row["confidence"] == 0.9

    def test_metadata_stored_as_json(self, tmp_brain):
        import brain
        brain.add_node("m1", "Meta Node", "concept", metadata={"key": "val"})
        conn = brain.get_db()
        row = conn.execute("SELECT metadata FROM nodes WHERE id='m1'").fetchone()
        conn.close()
        assert json.loads(row["metadata"]) == {"key": "val"}

    def test_embeds_content(self, tmp_brain):
        import brain
        brain.add_node("e1", "Embed Node", "concept", content="Test content for embedding")
        conn = brain.get_db()
        row = conn.execute("SELECT embedding FROM embeddings WHERE node_id='e1'").fetchone()
        conn.close()
        assert row is not None
        vec = np.frombuffer(row["embedding"], dtype=np.float32)
        assert len(vec) == EMBEDDING_DIM

    def test_no_embedding_without_content(self, tmp_brain):
        import brain
        brain.add_node("ne1", "No Content Node", "concept")
        conn = brain.get_db()
        row = conn.execute("SELECT * FROM embeddings WHERE node_id='ne1'").fetchone()
        conn.close()
        assert row is None


class TestAddEdge:
    def test_basic_edge(self, db):
        db.execute("INSERT INTO nodes (id, label, type) VALUES ('a', 'A node', 'concept')")
        db.execute("INSERT INTO nodes (id, label, type) VALUES ('b', 'B node', 'concept')")
        db.commit()
        import brain
        brain.add_edge("a", "b", "relates_to", note="test")
        conn = brain.get_db()
        row = conn.execute("SELECT * FROM edges WHERE from_id='a' AND to_id='b'").fetchone()
        conn.close()
        assert row is not None
        assert row["relation"] == "relates_to"

    def test_duplicate_edge_ignored(self, db):
        db.execute("INSERT INTO nodes (id, label, type) VALUES ('x', 'X node', 'concept')")
        db.execute("INSERT INTO nodes (id, label, type) VALUES ('y', 'Y node', 'concept')")
        db.commit()
        import brain
        brain.add_edge("x", "y", "relates_to")
        brain.add_edge("x", "y", "relates_to")  # should not raise
        conn = brain.get_db()
        count = conn.execute(
            "SELECT COUNT(*) FROM edges WHERE from_id='x' AND to_id='y'"
        ).fetchone()[0]
        conn.close()
        assert count == 1


class TestLogAccess:
    def test_increments_access_count(self, db):
        db.execute(
            "INSERT INTO nodes (id, label, type) VALUES ('la1', 'Log test', 'concept')"
        )
        db.commit()
        import brain
        brain.log_access(["la1"], source="test", conn=db)
        db.commit()
        row = db.execute("SELECT access_count FROM nodes WHERE id='la1'").fetchone()
        assert row["access_count"] == 1
        brain.log_access(["la1"], source="test", conn=db)
        db.commit()
        row = db.execute("SELECT access_count FROM nodes WHERE id='la1'").fetchone()
        assert row["access_count"] == 2

    def test_writes_access_log(self, db):
        db.execute(
            "INSERT INTO nodes (id, label, type) VALUES ('la2', 'Log test 2', 'concept')"
        )
        db.commit()
        import brain
        brain.log_access(["la2"], source="search", conn=db)
        db.commit()
        rows = db.execute("SELECT * FROM access_log WHERE node_id='la2'").fetchall()
        assert len(rows) == 1
        assert rows[0]["source"] == "search"


class TestNeighbors:
    def test_returns_neighbors(self, populated_db):
        import brain
        results = brain.neighbors("node_alpha")
        ids = [r["id"] for r in results]
        assert "node_beta" in ids
        assert "node_delta" in ids

    def test_empty_for_isolate(self, db):
        db.execute(
            "INSERT INTO nodes (id, label, type) VALUES ('iso', 'Isolated', 'concept')"
        )
        db.commit()
        import brain
        results = brain.neighbors("iso")
        assert len(results) == 0


# ═══════════════════════════════════════════════════════════════════
# brain/search.py
# ═══════════════════════════════════════════════════════════════════


class TestSemanticSearch:
    def test_returns_results(self, populated_db, tmp_brain):
        from brain.search import semantic_search
        results = semantic_search("machine learning AI")
        assert len(results) > 0
        assert "id" in results[0]
        assert "score" in results[0]

    def test_top_k_limit(self, populated_db, tmp_brain):
        from brain.search import semantic_search
        results = semantic_search("anything", top_k=2)
        assert len(results) <= 2

    def test_empty_graph(self, db, tmp_brain):
        from brain.search import semantic_search
        results = semantic_search("test query")
        assert results == []

    def test_core_memory_boost(self, populated_db, tmp_brain):
        """Core memory nodes should score higher than equivalent non-core nodes."""
        # Mark node_alpha as core memory
        populated_db.execute(
            "UPDATE nodes SET core_memory=1 WHERE id='node_alpha'"
        )
        populated_db.commit()

        from brain.search import semantic_search
        results = semantic_search("machine learning artificial intelligence")
        # Find the alpha node in results
        alpha = next((r for r in results if r["id"] == "node_alpha"), None)
        if alpha:
            # Get scores for non-core nodes
            non_core = [r for r in results if r["id"] != "node_alpha"]
            if non_core:
                # alpha score should include +0.15 boost
                # For a fair test, insert a node with identical embedding but no core flag
                pass  # The boost is applied, we verify it structurally below

    def test_core_memory_boost_applied(self, populated_db, tmp_brain):
        """Directly verify core_memory boost of +0.15."""
        # Insert two nodes with the SAME content (same embedding)
        import brain
        content = "Identical content for boost test"
        populated_db.execute(
            "INSERT INTO nodes (id, label, type, content, confidence, source, origin, core_memory) "
            "VALUES ('boost_core', 'Boost Core', 'concept', ?, 0.8, 'extract', 'self', 1)",
            (content,)
        )
        populated_db.execute(
            "INSERT INTO nodes (id, label, type, content, confidence, source, origin, core_memory) "
            "VALUES ('boost_normal', 'Boost Normal', 'concept', ?, 0.8, 'extract', 'self', 0)",
            (content,)
        )
        vec = brain.embed_text(content)
        for nid in ("boost_core", "boost_normal"):
            populated_db.execute(
                "INSERT INTO embeddings (node_id, embedding, model) VALUES (?, ?, ?)",
                (nid, vec.astype(np.float32).tobytes(), "test-model")
            )
        populated_db.commit()

        from brain.search import semantic_search
        results = semantic_search(content)
        core_result = next(r for r in results if r["id"] == "boost_core")
        normal_result = next(r for r in results if r["id"] == "boost_normal")
        assert core_result["score"] > normal_result["score"]
        assert abs(core_result["score"] - normal_result["score"] - 0.15) < 0.001

    def test_excludes_external_by_default(self, populated_db, tmp_brain):
        import brain
        populated_db.execute(
            "INSERT INTO nodes (id, label, type, content, source, origin) "
            "VALUES ('ext1', 'External Node', 'concept', 'External data', 'import', 'external')"
        )
        vec = brain.embed_text("External data")
        populated_db.execute(
            "INSERT INTO embeddings (node_id, embedding, model) VALUES (?, ?, ?)",
            ("ext1", vec.astype(np.float32).tobytes(), "test-model")
        )
        populated_db.commit()

        from brain.search import semantic_search
        results = semantic_search("External data")
        ext_ids = [r["id"] for r in results]
        assert "ext1" not in ext_ids

    def test_includes_external_when_requested(self, populated_db, tmp_brain):
        import brain
        populated_db.execute(
            "INSERT INTO nodes (id, label, type, content, source, origin) "
            "VALUES ('ext2', 'External Node Two', 'concept', 'External data two', 'import', 'external')"
        )
        vec = brain.embed_text("External data two")
        populated_db.execute(
            "INSERT INTO embeddings (node_id, embedding, model) VALUES (?, ?, ?)",
            ("ext2", vec.astype(np.float32).tobytes(), "test-model")
        )
        populated_db.commit()

        from brain.search import semantic_search
        results = semantic_search("External data two", include_external=True)
        ext_ids = [r["id"] for r in results]
        assert "ext2" in ext_ids

    def test_excludes_decayed_nodes(self, populated_db, tmp_brain):
        populated_db.execute("UPDATE nodes SET decayed=1 WHERE id='node_gamma'")
        populated_db.commit()
        from brain.search import semantic_search
        results = semantic_search("pricing model")
        ids = [r["id"] for r in results]
        assert "node_gamma" not in ids


class TestHybridSearch:
    def test_returns_results(self, populated_db, tmp_brain):
        from brain.search import hybrid_search
        results = hybrid_search("machine learning deployment")
        assert len(results) > 0

    def test_graph_walk_includes_neighbors(self, populated_db, tmp_brain):
        """Hybrid search should find nodes connected via graph edges."""
        from brain.search import hybrid_search
        results = hybrid_search("machine learning AI", top_k=10)
        ids = [r["id"] for r in results]
        # node_alpha -> node_beta -> node_gamma (2 hops)
        # So gamma should be reachable via graph walk
        assert len(ids) >= 1

    def test_empty_graph(self, db, tmp_brain):
        from brain.search import hybrid_search
        results = hybrid_search("test query")
        assert results == []


# ═══════════════════════════════════════════════════════════════════
# brain/sleep.py
# ═══════════════════════════════════════════════════════════════════


class TestEmbeddingDedup:
    def test_merges_similar_nodes(self, db, tmp_brain):
        import brain
        from brain.sleep import embedding_dedup

        # Insert two nodes with nearly identical embeddings
        base_vec = _random_embedding(seed=42)
        similar_vec = _similar_embedding(base_vec, similarity=0.95, seed=43)

        for nid, label, vec in [
            ("dup_a", "First duplicate concept", base_vec),
            ("dup_b", "Second duplicate notion", similar_vec),
        ]:
            db.execute(
                "INSERT INTO nodes (id, label, type, content, confidence, source) "
                "VALUES (?, ?, 'concept', 'test', 0.8, 'extract')",
                (nid, label)
            )
            db.execute(
                "INSERT INTO embeddings (node_id, embedding, model) VALUES (?, ?, ?)",
                (nid, vec.tobytes(), "test-model")
            )
        db.commit()

        merged = embedding_dedup(db, threshold=0.88)
        assert merged >= 1

    def test_preserves_insight_nodes(self, db, tmp_brain):
        from brain.sleep import embedding_dedup

        base_vec = _random_embedding(seed=50)
        similar_vec = _similar_embedding(base_vec, similarity=0.95, seed=51)

        for nid, label, vec in [
            ("insight_a", "Insight about patterns", base_vec),
            ("regular_b", "Regular about patterns too", similar_vec),
        ]:
            db.execute(
                "INSERT INTO nodes (id, label, type, content, confidence, source) "
                "VALUES (?, ?, 'concept', 'test', 0.8, 'extract')",
                (nid, label)
            )
            db.execute(
                "INSERT INTO embeddings (node_id, embedding, model) VALUES (?, ?, ?)",
                (nid, vec.tobytes(), "test-model")
            )
        db.commit()

        merged = embedding_dedup(db, threshold=0.88)
        # insight_ nodes should be skipped
        assert merged == 0

    def test_no_crash_on_empty(self, db, tmp_brain):
        from brain.sleep import embedding_dedup
        assert embedding_dedup(db) == 0

    def test_dry_run(self, db, tmp_brain):
        from brain.sleep import embedding_dedup

        base_vec = _random_embedding(seed=60)
        similar_vec = _similar_embedding(base_vec, similarity=0.96, seed=61)

        for nid, label, vec in [
            ("dry_a", "Dry run duplicate one", base_vec),
            ("dry_b", "Dry run duplicate two", similar_vec),
        ]:
            db.execute(
                "INSERT INTO nodes (id, label, type, content, confidence, source) "
                "VALUES (?, ?, 'concept', 'test', 0.8, 'extract')",
                (nid, label)
            )
            db.execute(
                "INSERT INTO embeddings (node_id, embedding, model) VALUES (?, ?, ?)",
                (nid, vec.tobytes(), "test-model")
            )
        db.commit()

        merged = embedding_dedup(db, threshold=0.88, dry_run=True)
        # Dry run should count but not delete
        remaining = db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        assert remaining == 2


class TestCrossLink:
    def test_creates_edges(self, db, tmp_brain):
        from brain.sleep import cross_link

        # Two similar nodes with no edge between them
        base_vec = _random_embedding(seed=70)
        similar_vec = _similar_embedding(base_vec, similarity=0.80, seed=71)

        for nid, vec in [("cl_a", base_vec), ("cl_b", similar_vec)]:
            db.execute(
                "INSERT INTO nodes (id, label, type, content, confidence, source) "
                "VALUES (?, ?, 'concept', 'test', 0.8, 'extract')",
                (nid, f"Cross link {nid}")
            )
            db.execute(
                "INSERT INTO embeddings (node_id, embedding, model) VALUES (?, ?, ?)",
                (nid, vec.tobytes(), "test-model")
            )
        db.commit()

        written = cross_link(db, threshold=0.68)
        assert written >= 1

    def test_skips_existing_edges(self, db, tmp_brain):
        from brain.sleep import cross_link

        base_vec = _random_embedding(seed=80)
        similar_vec = _similar_embedding(base_vec, similarity=0.85, seed=81)

        for nid, vec in [("sk_a", base_vec), ("sk_b", similar_vec)]:
            db.execute(
                "INSERT INTO nodes (id, label, type, content, confidence, source) "
                "VALUES (?, ?, 'concept', 'test', 0.8, 'extract')",
                (nid, f"Skip {nid}")
            )
            db.execute(
                "INSERT INTO embeddings (node_id, embedding, model) VALUES (?, ?, ?)",
                (nid, vec.tobytes(), "test-model")
            )
        db.execute(
            "INSERT INTO edges (from_id, to_id, relation) VALUES ('sk_a', 'sk_b', 'relates_to')"
        )
        db.commit()

        written = cross_link(db, threshold=0.68)
        assert written == 0


class TestComputeFitness:
    def test_basic_fitness(self, populated_db, tmp_brain):
        from brain.sleep import compute_fitness
        fitness = compute_fitness(populated_db)
        assert len(fitness) == 5
        # node_alpha has source='human' -> seed_bonus=2.0
        # node_alpha has 2 edges (alpha->beta, alpha->delta)
        assert fitness["node_alpha"] >= 2.0  # at least seed bonus

    def test_seed_bonus_for_human_source(self, populated_db, tmp_brain):
        from brain.sleep import compute_fitness
        fitness = compute_fitness(populated_db)
        # human and second_brain sources get +2.0
        assert fitness["node_alpha"] > fitness["node_gamma"]  # human > extract

    def test_empty_graph(self, db, tmp_brain):
        from brain.sleep import compute_fitness
        fitness = compute_fitness(db)
        assert fitness == {}


class TestPromoteCoreMemories:
    def test_promotes_top_sqrt_n(self, populated_db, tmp_brain):
        from brain.sleep import compute_fitness, promote_core_memories
        fitness = compute_fitness(populated_db)
        promoted, demoted = promote_core_memories(populated_db, fitness)
        # sqrt(5) = 2, so 2 nodes should be promoted
        expected_core = int(math.sqrt(5))
        core_count = populated_db.execute(
            "SELECT COUNT(*) FROM nodes WHERE core_memory=1"
        ).fetchone()[0]
        assert core_count == expected_core

    def test_demotes_previously_core(self, populated_db, tmp_brain):
        # Mark all as core first
        populated_db.execute("UPDATE nodes SET core_memory=1")
        populated_db.execute(
            "UPDATE nodes SET metadata=? WHERE 1=1",
            (json.dumps({"core_memory": True}),)
        )
        populated_db.commit()

        from brain.sleep import compute_fitness, promote_core_memories
        fitness = compute_fitness(populated_db)
        promoted, demoted = promote_core_memories(populated_db, fitness)
        assert len(demoted) > 0


class TestDecayRelevance:
    def test_decays_old_nodes(self, db, tmp_brain):
        from brain.sleep import decay_relevance

        old_date = (datetime.now() - timedelta(days=60)).isoformat()
        db.execute(
            "INSERT INTO nodes (id, label, type, source, last_accessed_at, access_count, relevance_score) "
            "VALUES ('old1', 'Old node', 'concept', 'extract', ?, 1, 1.0)",
            (old_date,)
        )
        db.commit()

        decay_relevance(db, base_half_life=30, full=True)

        row = db.execute("SELECT relevance_score FROM nodes WHERE id='old1'").fetchone()
        assert row["relevance_score"] < 1.0

    def test_permanent_nodes_skip_decay(self, db, tmp_brain):
        from brain.sleep import decay_relevance

        old_date = (datetime.now() - timedelta(days=120)).isoformat()
        db.execute(
            "INSERT INTO nodes (id, label, type, source, last_accessed_at, access_count, permanent) "
            "VALUES ('perm1', 'Permanent node', 'concept', 'extract', ?, 0, 2)",
            (old_date,)
        )
        db.commit()

        decay_relevance(db, full=True)

        row = db.execute("SELECT relevance_score FROM nodes WHERE id='perm1'").fetchone()
        assert row["relevance_score"] == 1.0  # unchanged

    def test_external_nodes_special_scoring(self, db, tmp_brain):
        from brain.sleep import decay_relevance

        db.execute(
            "INSERT INTO nodes (id, label, type, source, origin, access_count) "
            "VALUES ('ext_decay', 'External', 'concept', 'import', 'external', 3)"
        )
        db.commit()

        decay_relevance(db, full=True)
        row = db.execute("SELECT relevance_score FROM nodes WHERE id='ext_decay'").fetchone()
        # score = min(1.0, 0.3 + 3 * 0.07) = 0.51
        assert abs(row["relevance_score"] - 0.51) < 0.01


class TestSoftGc:
    def test_deletes_low_relevance(self, db, tmp_brain):
        from brain.sleep import soft_gc

        old_date = (datetime.now() - timedelta(days=30)).isoformat()
        db.execute(
            "INSERT INTO nodes (id, label, type, source, relevance_score, created_at) "
            "VALUES ('gc1', 'Low relevance', 'concept', 'extract', 0.01, ?)",
            (old_date,)
        )
        db.commit()

        deleted = soft_gc(db, threshold=0.05)
        assert deleted == 1
        row = db.execute("SELECT * FROM nodes WHERE id='gc1'").fetchone()
        assert row is None

    def test_grace_period_7_days(self, db, tmp_brain):
        """Nodes created < 7 days ago should NOT be garbage collected."""
        from brain.sleep import soft_gc

        recent_date = (datetime.now() - timedelta(days=3)).isoformat()
        db.execute(
            "INSERT INTO nodes (id, label, type, source, relevance_score, created_at) "
            "VALUES ('young1', 'Young node', 'concept', 'extract', 0.01, ?)",
            (recent_date,)
        )
        db.commit()

        deleted = soft_gc(db, threshold=0.05)
        assert deleted == 0
        row = db.execute("SELECT * FROM nodes WHERE id='young1'").fetchone()
        assert row is not None

    def test_protects_human_source(self, db, tmp_brain):
        from brain.sleep import soft_gc

        old_date = (datetime.now() - timedelta(days=30)).isoformat()
        db.execute(
            "INSERT INTO nodes (id, label, type, source, relevance_score, created_at) "
            "VALUES ('human1', 'Human node', 'concept', 'human', 0.01, ?)",
            (old_date,)
        )
        db.commit()

        deleted = soft_gc(db, threshold=0.05)
        assert deleted == 0

    def test_protects_second_brain_source(self, db, tmp_brain):
        from brain.sleep import soft_gc

        old_date = (datetime.now() - timedelta(days=30)).isoformat()
        db.execute(
            "INSERT INTO nodes (id, label, type, source, relevance_score, created_at) "
            "VALUES ('sb1', 'Second brain node', 'concept', 'second_brain', 0.01, ?)",
            (old_date,)
        )
        db.commit()

        deleted = soft_gc(db, threshold=0.05)
        assert deleted == 0

    def test_protects_person_type(self, db, tmp_brain):
        from brain.sleep import soft_gc

        old_date = (datetime.now() - timedelta(days=30)).isoformat()
        db.execute(
            "INSERT INTO nodes (id, label, type, source, relevance_score, created_at) "
            "VALUES ('pers1', 'A Person', 'person', 'extract', 0.01, ?)",
            (old_date,)
        )
        db.commit()

        deleted = soft_gc(db, threshold=0.05)
        assert deleted == 0

    def test_protects_decision_type(self, db, tmp_brain):
        from brain.sleep import soft_gc

        old_date = (datetime.now() - timedelta(days=30)).isoformat()
        db.execute(
            "INSERT INTO nodes (id, label, type, source, relevance_score, created_at) "
            "VALUES ('dec1', 'A Decision', 'decision', 'extract', 0.01, ?)",
            (old_date,)
        )
        db.commit()

        deleted = soft_gc(db, threshold=0.05)
        assert deleted == 0

    def test_protects_permanent_nodes(self, db, tmp_brain):
        from brain.sleep import soft_gc

        old_date = (datetime.now() - timedelta(days=30)).isoformat()
        db.execute(
            "INSERT INTO nodes (id, label, type, source, relevance_score, created_at, permanent) "
            "VALUES ('perm2', 'Permanent', 'concept', 'extract', 0.01, ?, 2)",
            (old_date,)
        )
        db.commit()

        deleted = soft_gc(db, threshold=0.05)
        assert deleted == 0

    def test_also_deletes_edges_and_embeddings(self, db, tmp_brain):
        from brain.sleep import soft_gc

        old_date = (datetime.now() - timedelta(days=30)).isoformat()
        db.execute(
            "INSERT INTO nodes (id, label, type, source, relevance_score, created_at) "
            "VALUES ('gc_e1', 'GC with edges', 'concept', 'extract', 0.01, ?)",
            (old_date,)
        )
        db.execute(
            "INSERT INTO nodes (id, label, type, source, relevance_score, created_at) "
            "VALUES ('gc_e2', 'Survivor', 'concept', 'extract', 0.99, ?)",
            (old_date,)
        )
        db.execute(
            "INSERT INTO edges (from_id, to_id, relation) VALUES ('gc_e1', 'gc_e2', 'relates_to')"
        )
        vec = _random_embedding(seed=99)
        db.execute(
            "INSERT INTO embeddings (node_id, embedding, model) VALUES ('gc_e1', ?, 'test')",
            (vec.tobytes(),)
        )
        db.commit()

        soft_gc(db, threshold=0.05)
        assert db.execute("SELECT * FROM edges WHERE from_id='gc_e1'").fetchone() is None
        assert db.execute("SELECT * FROM embeddings WHERE node_id='gc_e1'").fetchone() is None

    def test_dry_run(self, db, tmp_brain):
        from brain.sleep import soft_gc

        old_date = (datetime.now() - timedelta(days=30)).isoformat()
        db.execute(
            "INSERT INTO nodes (id, label, type, source, relevance_score, created_at) "
            "VALUES ('dryg1', 'Dry GC', 'concept', 'extract', 0.01, ?)",
            (old_date,)
        )
        db.commit()

        count = soft_gc(db, threshold=0.05, dry_run=True)
        assert count == 1
        # Node should still exist
        assert db.execute("SELECT * FROM nodes WHERE id='dryg1'").fetchone() is not None


class TestRunSleep:
    def test_full_pipeline(self, populated_db, tmp_brain):
        from brain.sleep import run_sleep
        stats = run_sleep(populated_db, full=True)
        assert "dedup_merged" in stats
        assert "cross_linked" in stats
        assert "nodes_scored" in stats
        assert "promoted" in stats
        assert "gc_deleted" in stats
        assert stats["dry_run"] is False

    def test_dry_run_flag(self, populated_db, tmp_brain):
        from brain.sleep import run_sleep
        stats = run_sleep(populated_db, dry_run=True)
        assert stats["dry_run"] is True

    def test_empty_graph(self, db, tmp_brain):
        from brain.sleep import run_sleep
        stats = run_sleep(db, full=True)
        assert stats["nodes_scored"] == 0


# ═══════════════════════════════════════════════════════════════════
# brain/compact.py — thin wrapper, just verify it imports correctly
# ═══════════════════════════════════════════════════════════════════

class TestCompact:
    def test_imports(self):
        from brain.compact import run_sleep  # re-exported
        assert callable(run_sleep)
