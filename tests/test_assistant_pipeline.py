"""
Integration and Unit Test Suite for Multi-Turn Conversational RAG Assistant.
Verifies Clarification Gate, Metadata Filtering, Grounded Checklist Citations, and Evals.
"""

import os
import pytest
from src.schemas import (
    ChatMessage,
    ConversationTurnDecision,
    GroundedAssistantResponse,
    ScoredDocument
)
from src.config import config, trace_collector, LocalTraceCollector
from src.knowledge_base import knowledge_base, CORPUS_DOCUMENTS
from src.assistant import assistant
from src.eval_harness import calculate_category_precision


@pytest.fixture(autouse=True)
def isolate_trace_collector(tmp_path, monkeypatch):
    """Ensure tests write traces to temporary storage without polluting benchmarks/langfuse_traces.json."""
    test_trace_file = str(tmp_path / "test_session_traces.json")
    monkeypatch.setattr(trace_collector, "trace_file_path", test_trace_file)
    monkeypatch.setattr(trace_collector, "traces", [])


def test_knowledge_base_corpus_integrity():
    """Verify domain expert and distractor documents are loaded with valid categories and metadata."""
    assert len(CORPUS_DOCUMENTS) >= 25
    valid_categories = {"health", "long_term_care", "auto", "life", "renters", "medicare"}
    for doc in CORPUS_DOCUMENTS:
        assert doc.category in valid_categories
        assert doc.doc_id.startswith("DOC-")
        assert len(doc.content) > 50


def test_qdrant_metadata_filtering_and_reranking():
    """Verify that metadata filtering strictly isolates documents and reranking ranks accurately."""
    results = knowledge_base.retrieve(
        query="Can I drop my spouse from my health policy?",
        category="health",
        top_k=3,
        use_filtering=True,
        use_reranking=True
    )
    assert len(results) > 0
    for doc in results:
        assert doc.category == "health"
        assert doc.doc_id.startswith("DOC-1")


def test_clarification_gate_on_underspecified_turn1():
    """Verify that ambiguous queries trigger needs_clarification=True and formulate a question."""
    history = [ChatMessage(role="user", content="Can I drop someone from my policy right now?")]
    decision = assistant.analyze_dialog_state(history)

    assert decision.needs_clarification is True
    assert decision.clarifying_question is not None
    assert len(decision.clarifying_question) > 10


def test_multi_turn_resolution_and_grounded_response():
    """Verify that providing missing context resolves the gate, filters retrieval, and cites documents."""
    turn1_user = "Can I drop someone from my policy right now?"
    turn1_bot = "Which type of insurance policy is this (e.g., health, auto, life)?"
    turn2_user = "It is an employer health insurance policy, we are not divorced, and open enrollment is closed."

    history = [
        ChatMessage(role="user", content=turn1_user),
        ChatMessage(role="assistant", content=turn1_bot),
        ChatMessage(role="user", content=turn2_user)
    ]

    res = assistant.handle_conversation_turn(history)

    assert res["type"] == "grounded_response"
    assert res["decision"].needs_clarification is False
    assert res["decision"].extracted_category == "health"

    grounded_resp = res["response"]
    assert isinstance(grounded_resp, GroundedAssistantResponse)
    assert len(grounded_resp.action_checklist) > 0

    # Ensure all cited sources exist in retrieved documents
    retrieved_ids = {d.doc_id for d in res["retrieved_docs"]}
    for item in grounded_resp.action_checklist:
        assert item.source_id in retrieved_ids
        assert len(item.action_title) > 0
        assert len(item.detailed_instruction) > 0


def test_trace_collector_logging(tmp_path):
    """Verify trace spans are recorded locally without raising network errors or mutating artifacts."""
    # LocalTraceCollector imported from src.config

    test_trace_file = str(tmp_path / "test_traces.json")
    collector = LocalTraceCollector(trace_file_path=test_trace_file)
    collector.record_trace({
        "span_name": "unit_test_trace",
        "status": "SUCCESS",
        "test_metric": 1.0
    })
    assert len(collector.traces) == 1
    assert os.path.isfile(test_trace_file)


def test_category_precision_metric():
    """Verify category precision accurately measures context purity and handles edge cases."""
    # calculate_category_precision and ScoredDocument imported at top

    # 1. Live vector index retrieval test
    docs_filtered = knowledge_base.retrieve(
        query="Can I drop my spouse from my health plan?",
        category="health",
        top_k=3,
        use_filtering=True
    )
    precision_filtered = calculate_category_precision(docs_filtered, expected_category="health", top_k=3)
    assert precision_filtered == 1.0

    # 2. Edge case: empty list returns 0.0
    assert calculate_category_precision([], expected_category="health", top_k=3) == 0.0

    # 3. Edge case: partial match and defensive case/whitespace normalization
    mock_docs = [
        ScoredDocument(doc_id="D1", title="T1", category="Health", sub_topic="Sub", content="C1", score=0.9),
        ScoredDocument(doc_id="D2", title="T2", category="auto", sub_topic="Sub", content="C2", score=0.8),
        ScoredDocument(doc_id="D3", title="T3", category="health ", sub_topic="Sub", content="C3", score=0.7),
    ]
    precision_mock = calculate_category_precision(mock_docs, expected_category="health", top_k=3)
    assert round(precision_mock, 4) == round(2 / 3, 4)

