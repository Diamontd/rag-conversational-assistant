"""
Multi-Turn Conversational RAG Assistant with Clarification Gate,
Metadata-Filtered Qdrant Retrieval, Grounded Checklist Synthesis, and Langfuse Observability.
"""

import os
import re
import json
import time
import logging
from typing import List, Dict, Any, Optional
from src.config import config, observe, trace_collector
from src.schemas import (
    ChatMessage,
    ConversationTurnDecision,
    GroundedAssistantResponse,
    ActionChecklistItem,
    ScoredDocument
)
from src.knowledge_base import knowledge_base

logger = logging.getLogger("rag_conversational_assistant.assistant")

# Robust domain category regex patterns (strictly matching user intent)
CATEGORY_PATTERNS = [
    ("long_term_care", [r"\blong-term care\b", r"\blong term care\b", r"\bltc\b", r"\badl\b", r"\badls\b", r"\bchronically ill\b"]),
    ("medicare", [r"\bmedicare\b", r"\bmedigap\b", r"\bpart a\b", r"\bpart b\b", r"\bpart c\b", r"\bpart d\b", r"\bmedicare ime\b", r"\bindirect medical education\b", r"\bhospital reimbursement\b"]),
    ("renters", [r"\brenter\b", r"\brenters\b", r"\brent\b", r"\btenant\b", r"\bapartment\b", r"\blandlord\b"]),
    ("auto", [r"\bauto\b", r"\bcar\b", r"\bdriver\b", r"\bvehicle\b", r"\bsplit limit\b", r"\bsplit limits\b", r"\bcrash\b", r"\bcollision\b"]),
    ("life", [r"\bwhole life\b", r"\bterm life\b", r"\bdeath benefit\b", r"\blife insurance\b", r"\blife policy beneficiary\b", r"\blife insurance beneficiary\b"]),
    ("health", [r"\bhealth insurance\b", r"\bhealth policy\b", r"\bhsa\b", r"\bcobra\b", r"\bhealth plan\b"])
]


def _call_llm_with_retry(messages: List[Dict[str, str]], max_retries: int = 2) -> Optional[str]:
    """Execute LLM chat completion with model failover and usage recording."""
    if not config.api_key or config.api_key == "placeholder_key_for_offline_mode":
        return None
    for model_name in config.fallback_models:
        for attempt in range(max_retries):
            try:
                response = config.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                usage = getattr(response, "usage", None)
                prompt_tokens = (getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
                completion_tokens = (getattr(usage, "completion_tokens", 0) or 0) if usage else 0
                estimated_cost_usd = round((prompt_tokens * 0.00000015) + (completion_tokens * 0.00000060), 6)
                trace_collector.record_llm_usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    model_name=model_name,
                    estimated_cost_usd=estimated_cost_usd
                )
                content = response.choices[0].message.content
                return content.strip() if content else None
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "quota" in err_str or "503" in err_str:
                    logger.warning(f"Model {model_name} rate limit or 503 ({e}), switching to next in cascade...")
                    break
    return None


class ConversationalRAGAssistant:
    """
    Production Multi-turn RAG Assistant enforcing clarification gates and grounded checklist generation.
    """
    def __init__(self):
        self.kb = knowledge_base

    @observe("analyze_dialog_state")
    def analyze_dialog_state(self, history: List[ChatMessage]) -> ConversationTurnDecision:
        """
        Analyze multi-turn dialog history to determine whether clarification is required,
        extract domain category, and rewrite self-contained search query.
        """
        formatted_dialog = "\n".join([f"{msg.role.upper()}: {msg.content}" for msg in history])

        prompt = f"""You are an expert dialog state tracker for a specialized insurance advisory assistant.
Supported categories are strictly: ["health", "long_term_care", "auto", "life", "renters", "medicare"].

Given the following conversation history:
\"\"\"
{formatted_dialog}
\"\"\"

Analyze whether the latest user request contains sufficient specific parameters to provide an accurate, safe answer.
CRITICAL RULES:
1. If the user query is ambiguous, generic, or lacks essential policy details (e.g. asking "Can I drop my spouse from my policy?" without stating whether it's health insurance or auto insurance; or asking "Are my insurance benefits taxed?" without specifying whether it's life insurance lump sum vs long-term care per-diem vs auto settlement), set "needs_clarification" to true.
2. If "needs_clarification" is true, provide an empathetic, concise "clarifying_question" asking specifically for the missing policy type and situation context. Set "search_query_rewritten" to the user query.
3. If the user has already provided sufficient context (e.g., they specified the insurance category and key circumstances across turns), set "needs_clarification" to false, leave "clarifying_question" as null, set "extracted_category" to one of the 6 valid categories, and formulate a comprehensive "search_query_rewritten" combining all user facts for vector search.

Respond ONLY with a valid JSON object matching this exact schema:
{{
  "needs_clarification": true/false,
  "clarifying_question": "string or null",
  "extracted_category": "health" | "long_term_care" | "auto" | "life" | "renters" | "medicare" | null,
  "search_query_rewritten": "self-contained search query string"
}}"""

        content = _call_llm_with_retry([
            {"role": "system", "content": "You are a precise JSON dialog state analyzer."},
            {"role": "user", "content": prompt}
        ])

        if content:
            try:
                data = json.loads(content)
                if not data.get("search_query_rewritten"):
                    data["search_query_rewritten"] = history[-1].content
                # Normalize extracted_category before schema validation
                if data.get("extracted_category") and isinstance(data["extracted_category"], str):
                    cat_clean = data["extracted_category"].strip().lower()
                    if cat_clean in {"health", "long_term_care", "auto", "life", "renters", "medicare"}:
                        data["extracted_category"] = cat_clean
                    else:
                        data["extracted_category"] = None
                return ConversationTurnDecision(**data)
            except Exception as e:
                logger.debug(f"JSON parsing error: {e}")

        # Resilient Deterministic Fallback Guardrail
        # Inspect ONLY user inputs to prevent prompt leakage
        user_turns = [m.content.lower() for m in history if m.role == "user"]
        combined_user_text = " ".join(user_turns)
        latest_user_text = user_turns[-1] if user_turns else ""

        detected_cat = None
        for cat, patterns in CATEGORY_PATTERNS:
            if any(re.search(p, combined_user_text) for p in patterns):
                detected_cat = cat
                break

        # Single-turn ambiguous queries lacking category specification -> Clarify
        if len(user_turns) <= 1 and detected_cat is None:
            return ConversationTurnDecision(
                needs_clarification=True,
                clarifying_question="Could you please specify which insurance category this relates to (e.g., health, auto, life, long-term care, renters, or Medicare)?",
                extracted_category=None,
                search_query_rewritten=latest_user_text
            )

        return ConversationTurnDecision(
            needs_clarification=False,
            clarifying_question=None,
            extracted_category=detected_cat,
            search_query_rewritten=combined_user_text
        )

    @observe("synthesize_grounded_response")
    def synthesize_grounded_response(
        self,
        query: str,
        retrieved_docs: List[ScoredDocument]
    ) -> GroundedAssistantResponse:
        """
        Synthesize a structured checklist strictly grounded in retrieved expert passages.
        """
        context_blocks = []
        for doc in retrieved_docs:
            context_blocks.append(
                f"[{doc.doc_id}] {doc.title}\nCategory: {doc.category} | Subtopic: {doc.sub_topic}\nContent: {doc.content}"
            )
        joined_context = "\n\n".join(context_blocks)

        prompt = f"""You are a certified insurance advisory assistant.
Synthesize a comprehensive, actionable response to the user query based ONLY on the provided expert documents.

Retrieved Context:
\"\"\"
{joined_context}
\"\"\"

User Query:
\"{query}\"

MANDATORY REQUIREMENTS:
1. Provide a concise 'situation_summary' summarizing the user's specific context.
2. Provide an 'action_checklist' of numbered steps. Each step MUST cite the exact 'source_id' (e.g. 'DOC-1001') from the retrieved context.
3. Do NOT invent document IDs. Only use document IDs present in the retrieved context.
4. List all unique cited document IDs in 'cited_source_ids'.

Output ONLY valid JSON matching this schema:
{{
  "situation_summary": "string",
  "action_checklist": [
    {{
      "step_number": 1,
      "action_title": "string",
      "detailed_instruction": "string",
      "source_id": "DOC-XXXX"
    }}
  ],
  "cited_source_ids": ["DOC-XXXX"]
}}"""

        content = _call_llm_with_retry([
            {"role": "system", "content": "You are a grounded insurance assistant outputting strict JSON."},
            {"role": "user", "content": prompt}
        ])

        if content:
            try:
                data = json.loads(content)
                return GroundedAssistantResponse(**data)
            except Exception as e:
                logger.debug(f"JSON parsing error in synthesis: {e}")

        # Deterministic Grounded Fallback
        cited_ids = [d.doc_id for d in retrieved_docs]
        checklist = []
        for i, doc in enumerate(retrieved_docs, start=1):
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', doc.content) if s.strip()]
            summary_sentences = " ".join(sentences[:2]) if len(sentences) >= 2 else (sentences[0] if sentences else doc.content)
            checklist.append(
                ActionChecklistItem(
                    step_number=i,
                    action_title=f"Review Policy Guidelines for {doc.title}",
                    detailed_instruction=f"According to verified documentation [{doc.doc_id}], {summary_sentences}",
                    source_id=doc.doc_id
                )
            )

        category_name = retrieved_docs[0].category if retrieved_docs else "general insurance"
        return GroundedAssistantResponse(
            situation_summary=f"Analysis of user inquiry regarding {category_name} policy provisions.",
            action_checklist=checklist,
            cited_source_ids=cited_ids
        )

    @observe("handle_conversation_turn")
    def handle_conversation_turn(self, history: List[ChatMessage]) -> Dict[str, Any]:
        """
        Execute full conversation turn:
        Turn 1: Clarification Gate -> returns clarification question if underspecified.
        Turn 2: Multi-turn synthesis -> filtered retrieval + reranking -> grounded response.
        """
        decision = self.analyze_dialog_state(history)

        if decision.needs_clarification:
            return {
                "type": "clarification",
                "decision": decision,
                "message": decision.clarifying_question,
                "retrieved_docs": []
            }

        retrieved_docs = self.kb.retrieve(
            query=decision.search_query_rewritten,
            category=decision.extracted_category,
            top_k=3,
            use_filtering=True,
            use_reranking=True
        )

        grounded_resp = self.synthesize_grounded_response(
            query=decision.search_query_rewritten,
            retrieved_docs=retrieved_docs
        )

        return {
            "type": "grounded_response",
            "decision": decision,
            "response": grounded_resp,
            "retrieved_docs": retrieved_docs
        }


# Global instance
assistant = ConversationalRAGAssistant()
