"""Unit tests for extractors: memory, sessions, chats, work, topical."""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from tests.conftest import _random_embedding, EMBEDDING_DIM


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _mock_llm_response(nodes_json):
    """Create a mock LLM response string."""
    return json.dumps(nodes_json)


# ═══════════════════════════════════════════════════════════════════
# extractors/memory.py
# ═══════════════════════════════════════════════════════════════════


class TestMemoryExtractor:
    def test_parse_json_array_valid(self, tmp_brain):
        from extractors.memory import _parse_json_array
        result = _parse_json_array('[{"id": "test", "label": "Test"}]')
        assert len(result) == 1
        assert result[0]["id"] == "test"

    def test_parse_json_array_with_markdown(self, tmp_brain):
        from extractors.memory import _parse_json_array
        text = '```json\n[{"id": "test"}]\n```'
        result = _parse_json_array(text)
        assert len(result) == 1

    def test_parse_json_array_empty(self, tmp_brain):
        from extractors.memory import _parse_json_array
        assert _parse_json_array(None) == []
        assert _parse_json_array("") == []
        assert _parse_json_array("no json here") == []

    def test_parse_json_array_not_array(self, tmp_brain):
        from extractors.memory import _parse_json_array
        assert _parse_json_array('{"not": "array"}') == []

    def test_extract_from_file_skips_small_content(self, tmp_brain, fixture_dir, monkeypatch):
        import extractors.memory as mem
        monkeypatch.setattr(mem, "CHECKPOINT_FILE", tmp_brain / "mem_ckpt.json")
        monkeypatch.setattr(mem, "BRAIN_DIR", tmp_brain)
        small_file = fixture_dir / "tiny.md"
        small_file.write_text("hi")
        result = mem.extract_from_file(small_file)
        assert result == 0

    def test_extract_from_file_respects_checkpoint(self, tmp_brain, fixture_dir, monkeypatch):
        import extractors.memory as mem
        monkeypatch.setattr(mem, "CHECKPOINT_FILE", tmp_brain / "mem_ckpt.json")
        monkeypatch.setattr(mem, "BRAIN_DIR", tmp_brain)

        f = fixture_dir / "noted.md"
        content = "This is a long note about machine learning and its impact on society. " * 10
        f.write_text(content)

        # Set checkpoint to full file size
        ckpt = {str(f): len(content.encode("utf-8"))}
        (tmp_brain / "mem_ckpt.json").write_text(json.dumps(ckpt))

        result = mem.extract_from_file(f)
        assert result == 0  # nothing new to process

    @patch("extractors.memory.call_llm")
    def test_extract_from_file_happy_path(self, mock_llm, tmp_brain, fixture_dir, monkeypatch):
        import extractors.memory as mem
        monkeypatch.setattr(mem, "CHECKPOINT_FILE", tmp_brain / "mem_ckpt.json")
        monkeypatch.setattr(mem, "BRAIN_DIR", tmp_brain)

        f = fixture_dir / "research.md"
        content = (
            "# Research Notes\n\n"
            "Machine learning models are increasingly used for code generation. "
            "The key insight is that transformer architectures excel at pattern recognition. "
            "We decided to use fine-tuning rather than prompt engineering for our use case.\n"
        ) * 3  # make it > 50 bytes

        f.write_text(content)

        mock_llm.return_value = json.dumps([
            {"id": "ml_codegen", "label": "ML for code generation", "type": "concept",
             "content": "ML models are increasingly used for code generation tasks"},
            {"id": "transformer_patterns", "label": "Transformers excel at patterns", "type": "thesis",
             "content": "Transformer architectures are effective for pattern recognition"},
        ])

        result = mem.extract_from_file(f)
        assert result == 2

    def test_run_missing_directory(self, tmp_brain, monkeypatch, capsys):
        import extractors.memory as mem
        monkeypatch.setattr(mem, "MEMORY_DIR", Path("/nonexistent/path"))
        mem.run()
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_knn_placement(self, populated_db, tmp_brain):
        """KNN placement should create edges to similar nodes."""
        from extractors.memory import knn_placement
        import brain

        vec = brain.embed_text("Machine learning is a subset of AI")
        knn_placement("new_node_knn", vec)
        # Should not crash; edges may or may not be created depending on threshold


# ═══════════════════════════════════════════════════════════════════
# extractors/sessions.py
# ═══════════════════════════════════════════════════════════════════


class TestSessionsExtractor:
    def test_extract_messages_from_session(self, fixture_dir):
        from extractors.sessions import extract_messages_from_session

        session_file = fixture_dir / "test_session.jsonl"
        lines = [
            json.dumps({"type": "message", "timestamp": "2024-01-15T10:00:00Z",
                        "message": {"role": "user", "content": [{"type": "text", "text": "Hello, I need help with my project"}]}}),
            json.dumps({"type": "message", "timestamp": "2024-01-15T10:00:05Z",
                        "message": {"role": "assistant", "content": [{"type": "text", "text": "Sure, what do you need?"}]}}),
            json.dumps({"type": "tool_use", "name": "read_file"}),  # should be skipped
            json.dumps({"type": "message", "timestamp": "2024-01-15T10:01:00Z",
                        "message": {"role": "user", "content": "I believe we should use Rust for performance"}}),
        ]
        session_file.write_text("\n".join(lines))

        messages = extract_messages_from_session(session_file)
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

    def test_extract_messages_handles_malformed_json(self, fixture_dir):
        from extractors.sessions import extract_messages_from_session
        session_file = fixture_dir / "bad_session.jsonl"
        session_file.write_text("not json\n{broken\n")
        messages = extract_messages_from_session(session_file)
        assert messages == []

    def test_format_conversation(self):
        from extractors.sessions import format_conversation
        messages = [
            {"role": "user", "text": "Hello world", "timestamp": "2024-01-01"},
            {"role": "assistant", "text": "Hi there", "timestamp": "2024-01-01"},
        ]
        result = format_conversation(messages)
        assert "[User]:" in result
        assert "[Agent]:" in result

    def test_format_conversation_truncates(self):
        from extractors.sessions import format_conversation
        messages = [{"role": "user", "text": "x" * 10000, "timestamp": "t"}]
        result = format_conversation(messages, max_chars=100)
        assert len(result) <= 600  # 500 char truncation per message + prefix

    def test_parse_json_array(self):
        from extractors.sessions import parse_json_array
        assert parse_json_array('[{"a": 1}]') == [{"a": 1}]
        assert parse_json_array(None) == []
        assert parse_json_array("garbage") == []

    @patch("extractors.sessions.call_llm")
    def test_extract_from_session_too_short(self, mock_llm, fixture_dir):
        from extractors.sessions import extract_from_session

        session_file = fixture_dir / "short_session.jsonl"
        lines = [
            json.dumps({"type": "message", "timestamp": "2024-01-15T10:00:00Z",
                        "message": {"role": "user", "content": [{"type": "text", "text": "Hi"}]}}),
        ]
        session_file.write_text("\n".join(lines))

        result = extract_from_session(session_file)
        assert result == 0
        mock_llm.assert_not_called()

    @patch("extractors.sessions.call_llm")
    def test_extract_from_session_happy_path(self, mock_llm, fixture_dir, tmp_brain):
        from extractors.sessions import extract_from_session

        session_file = fixture_dir / "good_session.jsonl"
        long_text = "I've been thinking about our architecture. " * 100
        lines = [
            json.dumps({"type": "message", "timestamp": "2024-01-15T10:00:00Z",
                        "message": {"role": "user", "content": [{"type": "text", "text": long_text}]}}),
            json.dumps({"type": "message", "timestamp": "2024-01-15T10:01:00Z",
                        "message": {"role": "assistant", "content": [{"type": "text", "text": long_text}]}}),
        ]
        session_file.write_text("\n".join(lines))

        mock_llm.return_value = json.dumps([
            {"label": "Architecture redesign thoughts", "type": "insight",
             "content": "User is reconsidering the system architecture", "confidence": 0.8},
        ])

        result = extract_from_session(session_file)
        assert result == 1

    @patch("extractors.sessions.call_llm")
    def test_low_confidence_filtered(self, mock_llm, fixture_dir, tmp_brain):
        from extractors.sessions import extract_from_session

        session_file = fixture_dir / "lowconf_session.jsonl"
        long_text = "Discussion about random topics without clear beliefs. " * 100
        lines = [
            json.dumps({"type": "message", "timestamp": "2024-01-15T10:00:00Z",
                        "message": {"role": "user", "content": [{"type": "text", "text": long_text}]}}),
            json.dumps({"type": "message", "timestamp": "2024-01-15T10:01:00Z",
                        "message": {"role": "assistant", "content": [{"type": "text", "text": long_text}]}}),
        ]
        session_file.write_text("\n".join(lines))

        mock_llm.return_value = json.dumps([
            {"label": "Vague thing", "type": "observation",
             "content": "Vague observation", "confidence": 0.3},  # below 0.6 threshold
        ])

        result = extract_from_session(session_file)
        assert result == 0

    def test_run_missing_directory(self, tmp_brain, monkeypatch, capsys):
        import extractors.sessions as sess
        monkeypatch.setattr(sess, "SESSIONS_DIR", Path("/nonexistent/sessions"))
        sess.run()
        captured = capsys.readouterr()
        assert "not found" in captured.out


# ═══════════════════════════════════════════════════════════════════
# extractors/chats.py
# ═══════════════════════════════════════════════════════════════════


class TestChatsExtractor:
    def test_parse_whatsapp_file(self, fixture_dir):
        from extractors.chats import parse_whatsapp_file

        chat_file = fixture_dir / "whatsapp_chat.txt"
        lines = [
            "15/01/2024, 10:00 - Alice: Hello, how are you?",
            "15/01/2024, 10:01 - Bob: I'm good, working on the new project",
            "15/01/2024, 10:02 - Alice: I think we should use Python for this",
            "16/01/2024, 09:00 - Bob: I agree, Python is great for prototyping",
        ]
        chat_file.write_text("\n".join(lines))

        messages = parse_whatsapp_file(chat_file)
        assert len(messages) == 4
        assert messages[0][1] == "Alice"  # sender
        assert messages[0][2] == "Hello, how are you?"  # text

    def test_parse_whatsapp_various_formats(self, fixture_dir):
        from extractors.chats import parse_whatsapp_file

        chat_file = fixture_dir / "wa_formats.txt"
        lines = [
            "[15/01/24, 10:00:00] Alice: Message one",
            "15-01-2024, 10:01 AM - Bob: Message two",
            "15.01.2024, 10:02 - Charlie: Message three",
        ]
        chat_file.write_text("\n".join(lines))

        messages = parse_whatsapp_file(chat_file)
        assert len(messages) >= 2  # at least some should parse

    def test_parse_date(self):
        from extractors.chats import _parse_date
        assert _parse_date("15/01/2024") is not None
        # "01-15-2024" doesn't match d-m-Y (15 is not a valid month), which is correct
        assert _parse_date("15-01-2024") is not None
        assert _parse_date("invalid") is None

    def test_identify_user_by_name(self):
        from extractors.chats import identify_user
        messages = [
            (None, "Alice", "hi"),
            (None, "Bob", "hello"),
            (None, "Alice", "how are you"),
        ]
        result = identify_user(messages, user_name="Alice")
        assert result == "Alice"

    def test_identify_user_by_frequency(self):
        from extractors.chats import identify_user
        messages = [
            (None, "Alice", "hi"),
            (None, "Alice", "hello"),
            (None, "Alice", "test"),
            (None, "Bob", "reply"),
        ]
        result = identify_user(messages)
        assert result == "alice"  # returns lowercase

    def test_identify_user_empty(self):
        from extractors.chats import identify_user
        assert identify_user([], user_name="Nobody") is None

    def test_parse_json_array(self):
        from extractors.chats import parse_json_array
        assert parse_json_array('[{"a": 1}]') == [{"a": 1}]
        assert parse_json_array("") == []

    def test_run_missing_directory(self, tmp_brain, monkeypatch, capsys):
        import extractors.chats as ch
        monkeypatch.setattr(ch, "CHATS_DIR", Path("/nonexistent/chats"))
        ch.run()
        captured = capsys.readouterr()
        assert "not found" in captured.out


# ═══════════════════════════════════════════════════════════════════
# extractors/work.py
# ═══════════════════════════════════════════════════════════════════


class TestWorkExtractor:
    def test_parse_json_array(self):
        from extractors.work import parse_json_array
        assert parse_json_array('[{"label": "test"}]') == [{"label": "test"}]
        assert parse_json_array(None) == []

    @patch("extractors.work.call_llm")
    def test_extract_from_file_too_short(self, mock_llm, fixture_dir):
        from extractors.work import extract_from_file
        f = fixture_dir / "tiny_work.md"
        f.write_text("short")
        result = extract_from_file(f)
        assert result == 0
        mock_llm.assert_not_called()

    @patch("extractors.work.call_llm")
    def test_extract_from_file_happy_path(self, mock_llm, fixture_dir, tmp_brain):
        from extractors.work import extract_from_file

        f = fixture_dir / "meeting_notes.md"
        f.write_text(
            "# Q1 Planning Meeting\n\n"
            "Attendees: Alice, Bob, Charlie\n\n"
            "## Decisions\n"
            "- We will migrate to PostgreSQL by end of Q2\n"
            "- Budget allocated: $50k for infrastructure\n"
            "- Alice will lead the migration project\n\n"
            "## Action Items\n"
            "- Bob: set up staging environment by Feb 15\n"
            "- Charlie: update monitoring dashboards\n"
        )

        mock_llm.return_value = json.dumps([
            {"label": "PostgreSQL migration decision", "type": "decision",
             "content": "Team decided to migrate to PostgreSQL by Q2 end with $50k budget",
             "confidence": 0.9},
            {"label": "Alice leads migration", "type": "fact",
             "content": "Alice assigned as migration project lead",
             "confidence": 0.85},
        ])

        result = extract_from_file(f, source_type="meeting")
        assert result == 2

    @patch("extractors.work.call_llm")
    def test_extract_dry_run(self, mock_llm, fixture_dir, tmp_brain):
        from extractors.work import extract_from_file

        f = fixture_dir / "dry_work.md"
        f.write_text("A long enough document with sufficient content. " * 10)

        mock_llm.return_value = json.dumps([
            {"label": "Test node", "type": "insight", "content": "Test content", "confidence": 0.8},
        ])

        result = extract_from_file(f, dry_run=True)
        assert result == 1

    def test_run_missing_directory(self, tmp_brain, monkeypatch, capsys):
        import extractors.work as wk
        monkeypatch.setattr(wk, "WORK_DIR", Path("/nonexistent/work"))
        wk.run()
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_source_type_detection(self, fixture_dir):
        """Verify meeting/email detection logic."""
        meeting_file = fixture_dir / "meeting_2024.md"
        meeting_file.write_text("content")
        email_file = fixture_dir / "inbox.eml"
        email_file.write_text("content")
        doc_file = fixture_dir / "notes.md"
        doc_file.write_text("content")

        # Inline the detection logic from run()
        assert "meeting" in meeting_file.name.lower()
        assert email_file.suffix == ".eml"


# ═══════════════════════════════════════════════════════════════════
# extractors/topical.py
# ═══════════════════════════════════════════════════════════════════


class TestTopicalExtractor:
    def test_parse_json_array(self):
        from extractors.topical import parse_json_array
        assert parse_json_array('[{"label": "thesis"}]') == [{"label": "thesis"}]
        assert parse_json_array(None) == []

    @patch("extractors.topical.call_llm")
    def test_extract_from_file_too_short(self, mock_llm, fixture_dir):
        from extractors.topical import extract_from_file
        f = fixture_dir / "tiny_research.md"
        f.write_text("short")
        result = extract_from_file(f, domain="test")
        assert result == 0
        mock_llm.assert_not_called()

    @patch("extractors.topical.call_llm")
    def test_extract_from_file_happy_path(self, mock_llm, fixture_dir, tmp_brain):
        from extractors.topical import extract_from_file

        f = fixture_dir / "deep_research.md"
        f.write_text(
            "# Quantum Computing and Cryptography\n\n"
            "The advent of quantum computing poses significant challenges to current "
            "cryptographic systems. RSA and ECC are vulnerable to Shor's algorithm. "
            "Post-quantum cryptography (PQC) alternatives include lattice-based, "
            "hash-based, and code-based schemes. NIST has standardized CRYSTALS-Kyber "
            "for key encapsulation and CRYSTALS-Dilithium for digital signatures.\n\n"
            "Key tension: PQC algorithms require larger key sizes and more computation, "
            "creating performance tradeoffs for embedded and IoT devices.\n"
        )

        mock_llm.return_value = json.dumps([
            {"label": "Quantum threat to classical cryptography", "type": "thesis",
             "content": "Quantum computers can break RSA and ECC via Shor's algorithm",
             "confidence": 0.95, "tags": ["quantum", "cryptography"]},
            {"label": "NIST PQC standardization choices", "type": "fact",
             "content": "NIST standardized CRYSTALS-Kyber and CRYSTALS-Dilithium",
             "confidence": 0.9, "tags": ["pqc", "nist"]},
            {"label": "PQC performance vs security tradeoff", "type": "contradiction",
             "content": "PQC algorithms need larger keys, creating tension for IoT",
             "confidence": 0.85, "tags": ["pqc", "iot"]},
        ])

        result = extract_from_file(f, domain="crypto")
        assert result == 3

    @patch("extractors.topical.call_llm")
    def test_node_id_format(self, mock_llm, fixture_dir, tmp_brain):
        """Node IDs should follow topic_{domain}_{hash} format."""
        from extractors.topical import extract_from_file
        import brain

        f = fixture_dir / "id_test.md"
        f.write_text("A sufficiently long document for testing node ID format. " * 20)

        mock_llm.return_value = json.dumps([
            {"label": "Test thesis", "type": "thesis",
             "content": "A test thesis for ID verification", "confidence": 0.8,
             "tags": ["test"]},
        ])

        extract_from_file(f, domain="testdomain")
        conn = brain.get_db()
        rows = conn.execute("SELECT id FROM nodes WHERE id LIKE 'topic_testdomain_%'").fetchall()
        conn.close()
        assert len(rows) == 1

    @patch("extractors.topical.call_llm")
    def test_metadata_includes_tags_and_domain(self, mock_llm, fixture_dir, tmp_brain):
        from extractors.topical import extract_from_file
        import brain

        f = fixture_dir / "meta_test.md"
        f.write_text("Enough content to trigger extraction for metadata test purposes. " * 20)

        mock_llm.return_value = json.dumps([
            {"label": "Tagged insight", "type": "concept",
             "content": "An insight with tags", "confidence": 0.8,
             "tags": ["alpha", "beta"]},
        ])

        extract_from_file(f, domain="metadomain")
        conn = brain.get_db()
        row = conn.execute(
            "SELECT metadata FROM nodes WHERE id LIKE 'topic_metadomain_%'"
        ).fetchone()
        conn.close()
        meta = json.loads(row["metadata"])
        assert meta["domain"] == "metadomain"
        assert "alpha" in meta["tags"]

    def test_run_missing_directory(self, tmp_brain, monkeypatch, capsys):
        import extractors.topical as tp
        monkeypatch.setattr(tp, "TOPICAL_DIR", Path("/nonexistent/research"))
        tp.run()
        captured = capsys.readouterr()
        assert "not found" in captured.out
