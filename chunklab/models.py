"""Core data models for chunklab."""

from typing import Literal

from pydantic import BaseModel, Field

ElementType = Literal["heading", "paragraph", "table", "list", "code"]


class Element(BaseModel):
    """A lightweight structural element extracted by a loader."""

    type: ElementType
    text: str
    char_span: tuple[int, int]
    level: int | None = None


class Document(BaseModel):
    id: str
    source_path: str
    text: str
    elements: list[Element] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class Chunk(BaseModel):
    id: str
    doc_id: str
    text: str
    token_count: int
    char_span: tuple[int, int]
    strategy: str
    contains_table: bool = False
    section_path: list[str] = Field(default_factory=list)


class Question(BaseModel):
    id: str
    query: str
    gold_snippets: list[str] = Field(default_factory=list)
    gold_answer: str | None = None
    tags: list[str] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    rank: int
    is_hit: bool = False


class QuestionResult(BaseModel):
    question_id: str
    strategy: str
    retrieved: list[RetrievedChunk]
    hit: bool
    first_hit_rank: int | None = None
    split_across_chunks: bool = False
    gold_found_count: int = 0
    gold_total: int = 0
    found_gold_indices: list[int] = Field(default_factory=list)


class ChunkHealth(BaseModel):
    num_chunks: int
    tokens_min: int
    tokens_median: float
    tokens_mean: float
    tokens_max: int
    pct_tiny: float
    pct_oversized: float
    boundary_health: float
    table_integrity: float | None = None
    token_histogram: list[tuple[int, int]] = Field(default_factory=list)  # (bucket_start, count)


class StrategyResult(BaseModel):
    strategy: str
    config: dict = Field(default_factory=dict)
    recall_at_k: float
    hit_rate_at_k: float
    mrr: float
    precision_at_k: float
    retrieved_tokens_at_k: float = 0.0  # mean tokens handed to the LLM per question
    context_efficiency: float = 0.0  # found-gold tokens / retrieved tokens (mean)
    balanced_score: float = 0.0  # recall penalized by relative context cost
    ci95: tuple[float, float] | None = None  # bootstrap CI of the primary metric
    chunk_health: ChunkHealth
    per_question: list[QuestionResult] = Field(default_factory=list)


class EvalReport(BaseModel):
    schema_version: str = "1.1"  # public JSON contract; see docs/schema.md
    corpus_summary: dict = Field(default_factory=dict)
    strategy_results: list[StrategyResult] = Field(default_factory=list)  # ranked best-first
    recommendation: str = ""
    warnings: list[str] = Field(default_factory=list)
    generated_at: str = ""
    viz: dict | None = None  # chunk-boundary visualization for a sample document
