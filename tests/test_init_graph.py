"""init_graph and add_edge tests."""
import sys, os, sqlite3, shutil, uuid
sys.path.insert(0, os.path.expanduser("~/.openclaw/workspace/graph"))
import pytest

BRAIN_DB = os.path.expanduser("~/.openclaw/workspace/graph/brain.db")
TEST_DB = "/tmp/test_brain_init.db"


@pytest.fixture(scope="module")
def test_conn():
    shutil.copy2(BRAIN_DB, TEST_DB)
    conn = sqlite3.connect(TEST_DB)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


def test_add_edge_no_relation(test_conn):
    """INSERT into edges without relation column should work."""
    rows = test_conn.execute("SELECT id FROM nodes WHERE id IS NOT NULL LIMIT 2").fetchall()
    if len(rows) < 2:
        pytest.skip("Not enough nodes")
    # row_factory = Row, access by column name
    from_id = rows[0]["id"]
    to_id = rows[1]["id"]
    if not from_id or not to_id:
        pytest.skip("Node IDs are null")
    test_conn.execute("DELETE FROM edges WHERE from_id = ? AND to_id = ?", (from_id, to_id))
    test_conn.commit()
    test_conn.execute(
        "INSERT OR IGNORE INTO edges (from_id, to_id, weight, note, source) VALUES (?,?,?,?,?)",
        (from_id, to_id, 0.8, "test edge", "test")
    )
    test_conn.commit()
    count = test_conn.execute(
        "SELECT COUNT(*) FROM edges WHERE from_id=? AND to_id=?", (from_id, to_id)
    ).fetchone()[0]
    assert count == 1


def test_add_edge_duplicate_no_raise(test_conn):
    """Duplicate (from_id, to_id) should be silently ignored (INSERT OR IGNORE)."""
    rows = test_conn.execute("SELECT from_id, to_id FROM edges LIMIT 1").fetchone()
    if rows is None:
        pytest.skip("No edges in test db")
    # Try inserting same pair again — should not raise
    test_conn.execute(
        "INSERT OR IGNORE INTO edges (from_id, to_id) VALUES (?, ?)",
        (rows["from_id"], rows["to_id"])
    )
    test_conn.commit()


def test_edges_no_relation_column(test_conn):
    cols = [row[1] for row in test_conn.execute("PRAGMA table_info(edges)").fetchall()]
    assert "relation" not in cols
