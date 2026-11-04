"""
Golden Dataset Evaluation Pipeline and Production Metrics Harness.
Computes Clarification Gate Accuracy, Unfiltered vs. Filtered Recall@3 & MRR@3,
Citation Faithfulness, Latency (p50, p95), and writes benchmarks/golden_eval_results.csv.
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from src.config import config, trace_collector, observe
from src.schemas import (
    ChatMessage,
    GoldenEvalScenario,
    ScoredDocument
)
from src.assistant import assistant
from src.knowledge_base import knowledge_base

logger = logging.getLogger("rag_conversational_assistant.eval_harness")


def calculate_retrieval_metrics(retrieved_docs: List[ScoredDocument], target_doc_ids: List[str], top_k: int = 3):
    """Calculate Recall@K and MRR@K for retrieved documents."""
    retrieved_ids = [d.doc_id for d in retrieved_docs[:top_k]]
    target_set = set(target_doc_ids)

    # Hits in top K
    hits = [doc_id for doc_id in retrieved_ids if doc_id in target_set]
    recall = len(hits) / max(1, len(target_set))

    # Reciprocal Rank of first relevant document
    mrr = 0.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in target_set:
            mrr = 1.0 / rank
            break

    return recall, mrr


def calculate_category_precision(retrieved_docs: List[ScoredDocument], expected_category: str, top_k: int = 3) -> float:
    """Calculate the fraction of top-K retrieved documents matching the expected category (Context Purity)."""
    docs_k = retrieved_docs[:top_k]
    if not docs_k:
        return 0.0
    norm_expected = (expected_category or "").strip().lower()
    matching = sum(1 for d in docs_k if (d.category or "").strip().lower() == norm_expected)
    return matching / len(docs_k)


@observe("run_golden_eval_suite")
def run_evaluation_suite(
    golden_dataset_path: Optional[str] = None,
    output_csv_path: Optional[str] = None,
    force_rerun: bool = True
) -> pd.DataFrame:
    """
    Run all Golden Dataset scenarios through live assistant and compute production benchmark metrics.
    Supports resuming from existing CSV snapshots unless force_rerun is True.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if golden_dataset_path is None:
        golden_dataset_path = os.path.join(base_dir, "data", "golden_dataset.json")
    if output_csv_path is None:
        output_csv_path = os.path.join(base_dir, "benchmarks", "golden_eval_results.csv")

    with open(golden_dataset_path, "r", encoding="utf-8") as f:
        raw_scenarios = json.load(f)

    scenarios = [GoldenEvalScenario(**s) for s in raw_scenarios]
    print(f"Loaded {len(scenarios)} Golden Dataset evaluation scenarios.")

    if force_rerun:
        trace_collector.clear()

    # Check for existing completed scenarios unless force_rerun is True
    results_records: List[Dict[str, Any]] = []
    completed_ids = set()
    if not force_rerun and os.path.isfile(output_csv_path):
        try:
            existing_df = pd.read_csv(output_csv_path)
            results_records = existing_df.to_dict(orient="records")
            completed_ids = {r["scenario_id"] for r in results_records}
            print(f"Found existing progress: {len(completed_ids)}/{len(scenarios)} scenarios completed.")
        except Exception as e:
            logger.debug(f"Could not read existing progress: {e}")

    for idx, scen in enumerate(scenarios, start=1):
        if scen.scenario_id in completed_ids:
            print(f"[{idx}/{len(scenarios)}] Skipping {scen.scenario_id} (already evaluated).")
            continue

        print(f"[{idx}/{len(scenarios)}] Evaluating {scen.scenario_id}: {scen.title}...")
        start_trace_idx = len(trace_collector.traces)

        # TURN 1: Ambiguous Query -> Must trigger Clarification Gate
        t1_start = time.time()
        t1_history = [ChatMessage(role="user", content=scen.turn1_user)]
        t1_res = assistant.handle_conversation_turn(t1_history)
        t1_latency_ms = round((time.time() - t1_start) * 1000, 2)

        t1_clarification_success = (t1_res["type"] == "clarification" and t1_res["decision"].needs_clarification is True)
        clarifying_question = t1_res.get("message") or "Which insurance category does this apply to?"

        # TURN 2: User provides missing context -> Must answer with Grounded Checklist
        t2_start = time.time()
        t2_history = [
            ChatMessage(role="user", content=scen.turn1_user),
            ChatMessage(role="assistant", content=clarifying_question),
            ChatMessage(role="user", content=scen.turn2_user)
        ]
        t2_res = assistant.handle_conversation_turn(t2_history)
        t2_latency_ms = round((time.time() - t2_start) * 1000, 2)

        t2_answer_success = (t2_res["type"] == "grounded_response" and t2_res["decision"].needs_clarification is False)
        t2_category_match = (t2_res["decision"].extracted_category == scen.expected_category)

        # Baseline Retrieval: Unfiltered Single-Turn Vector Search on initial ambiguous query Turn 1
        unfiltered_docs = knowledge_base.retrieve(
            query=scen.turn1_user,
            category=None,
            top_k=3,
            use_filtering=False,
            use_reranking=False
        )
        unfiltered_recall, unfiltered_mrr = calculate_retrieval_metrics(unfiltered_docs, scen.target_doc_ids, top_k=3)
        unfiltered_cat_precision = calculate_category_precision(unfiltered_docs, scen.expected_category, top_k=3)

        # Multi-Turn Clarified + Metadata Filtered & Reranked Retrieval (evaluated on Turn 2)
        filtered_docs = t2_res.get("retrieved_docs", [])
        filtered_recall, filtered_mrr = calculate_retrieval_metrics(filtered_docs, scen.target_doc_ids, top_k=3)
        filtered_cat_precision = calculate_category_precision(filtered_docs, scen.expected_category, top_k=3)

        # Citation Faithfulness / Grounding Rate
        grounded_response = t2_res.get("response")
        valid_citations = 0
        total_items = 0
        retrieved_id_set = {d.doc_id for d in filtered_docs}

        if grounded_response and hasattr(grounded_response, "action_checklist"):
            total_items = len(grounded_response.action_checklist)
            for item in grounded_response.action_checklist:
                # Check if cited source ID belongs to valid retrieved documents
                if item.source_id in retrieved_id_set:
                    valid_citations += 1

        citation_faithfulness = (valid_citations / max(1, total_items)) * 100.0

        results_records.append({
            "scenario_id": scen.scenario_id,
            "title": scen.title,
            "expected_category": scen.expected_category,
            "extracted_category": t2_res["decision"].extracted_category,
            "category_match": t2_category_match,
            "t1_clarification_accuracy": 1.0 if t1_clarification_success else 0.0,
            "t2_answer_accuracy": 1.0 if t2_answer_success else 0.0,
            "unfiltered_recall_at_3": round(unfiltered_recall, 4),
            "unfiltered_mrr_at_3": round(unfiltered_mrr, 4),
            "unfiltered_category_precision_at_3": round(unfiltered_cat_precision, 4),
            "filtered_recall_at_3": round(filtered_recall, 4),
            "filtered_mrr_at_3": round(filtered_mrr, 4),
            "filtered_category_precision_at_3": round(filtered_cat_precision, 4),
            "citation_faithfulness_pct": round(citation_faithfulness, 2),
            "checklist_items_count": total_items,
            "valid_citations_count": valid_citations,
            "t1_latency_ms": t1_latency_ms,
            "t2_latency_ms": t2_latency_ms,
            "total_latency_ms": round(t1_latency_ms + t2_latency_ms, 2),
            "total_tokens": sum(t.get("total_tokens", 0) for t in trace_collector.traces[start_trace_idx:]),
            "estimated_cost_usd": round(sum(t.get("estimated_cost_usd", 0.0) for t in trace_collector.traces[start_trace_idx:]), 6)
        })

        # Save progressive CSV snapshot
        df_prog = pd.DataFrame(results_records)
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df_prog.to_csv(output_csv_path, index=False)

    df = pd.DataFrame(results_records)
    df.to_csv(output_csv_path, index=False)
    print(f"Successfully saved evaluation results to {output_csv_path}")

    # Log summary statistics
    print("\n" + "=" * 60)
    print("GOLDEN DATASET EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Scenarios Evaluated: {len(df)}")
    print(f"Clarification Gate Accuracy (Turn 1): {df['t1_clarification_accuracy'].mean() * 100:.1f}%")
    print(f"Category Extraction Accuracy: {df['category_match'].mean() * 100:.1f}%")
    print(f"Unfiltered Retrieval Recall@3: {df['unfiltered_recall_at_3'].mean():.4f}")
    print(f"Filtered + Reranked Recall@3:  {df['filtered_recall_at_3'].mean():.4f} (+{(df['filtered_recall_at_3'].mean() - df['unfiltered_recall_at_3'].mean()):.4f})")
    print(f"Unfiltered Retrieval MRR@3:    {df['unfiltered_mrr_at_3'].mean():.4f}")
    print(f"Filtered + Reranked MRR@3:     {df['filtered_mrr_at_3'].mean():.4f} (+{(df['filtered_mrr_at_3'].mean() - df['unfiltered_mrr_at_3'].mean()):.4f})")
    print(f"Unfiltered Category Precision@3: {df['unfiltered_category_precision_at_3'].mean():.4f}")
    print(f"Filtered Category Precision@3:   {df['filtered_category_precision_at_3'].mean():.4f} (+{(df['filtered_category_precision_at_3'].mean() - df['unfiltered_category_precision_at_3'].mean()):.4f})")
    print(f"Citation Faithfulness Rate:    {df['citation_faithfulness_pct'].mean():.1f}%")
    print(f"Latency Turn 1 (p50 / p95):    {df['t1_latency_ms'].quantile(0.5):.1f} ms / {df['t1_latency_ms'].quantile(0.95):.1f} ms")
    print(f"Latency Turn 2 (p50 / p95):    {df['t2_latency_ms'].quantile(0.5):.1f} ms / {df['t2_latency_ms'].quantile(0.95):.1f} ms")
    avg_tokens = df["total_tokens"].mean() if "total_tokens" in df else 0
    avg_cost = df["estimated_cost_usd"].mean() if "estimated_cost_usd" in df else 0
    print(f"Average Tokens / Scenario:     {avg_tokens:.1f}")
    print(f"Average Est. Cost / Scenario:  ${avg_cost:.6f}")
    print("=" * 60)

    # Persist Langfuse trace spans
    trace_collector.save()
    return df


if __name__ == "__main__":
    run_evaluation_suite()
