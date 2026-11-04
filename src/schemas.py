"""
Pydantic v2 Contracts for Multi-Turn Conversational RAG State, Checklists & Evals.
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ConversationTurnDecision(BaseModel):
    needs_clarification: bool = Field(
        description="True if the user query lacks critical context (e.g., policy category, dependent role, or state/coverage tier) required to give an accurate answer."
    )
    clarifying_question: Optional[str] = Field(
        default=None,
        description="A concise, empathetic clarifying question asking only for the missing context when needs_clarification is True."
    )
    extracted_category: Optional[Literal["health", "long_term_care", "auto", "life", "renters", "medicare"]] = Field(
        default=None,
        description="The domain category extracted from the conversation history for Qdrant metadata filtering."
    )
    search_query_rewritten: str = Field(
        default="",
        description="Self-contained search query combining multi-turn user context for vector retrieval."
    )


class ActionChecklistItem(BaseModel):
    step_number: int
    action_title: str
    detailed_instruction: str
    source_id: str = Field(description="Exact document ID from the retrieved context, e.g., DOC-3334.")


class GroundedAssistantResponse(BaseModel):
    situation_summary: str = Field(description="Brief synthesis of the user's specific situation.")
    action_checklist: List[ActionChecklistItem] = Field(
        description="Step-by-step actionable checklist grounded strictly in the retrieved expert materials."
    )
    cited_source_ids: List[str] = Field(description="List of unique [DOC-ID] identifiers used.")


class DocumentChunk(BaseModel):
    doc_id: str
    title: str
    category: Literal["health", "long_term_care", "auto", "life", "renters", "medicare"]
    sub_topic: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ScoredDocument(BaseModel):
    doc_id: str
    title: str
    category: str
    sub_topic: str
    content: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GoldenEvalScenario(BaseModel):
    scenario_id: str
    title: str
    turn1_user: str
    turn2_user: str
    expected_category: Literal["health", "long_term_care", "auto", "life", "renters", "medicare"]
    target_doc_ids: List[str]
    required_key_facts: List[str]
