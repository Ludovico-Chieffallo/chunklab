"""Config schema, YAML loading, and defaults."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from chunklab.models import Question


class EmbeddingConfig(BaseModel):
    backend: Literal["local", "openai", "fake"] = "local"
    model: str = "BAAI/bge-small-en-v1.5"


class RetrievalConfig(BaseModel):
    mode: Literal["dense", "bm25", "hybrid"] = "dense"
    top_k: int = Field(5, ge=1)


class EvalConfig(BaseModel):
    fuzzy_threshold: float = Field(0.90, ge=0.0, le=1.0)
    ranking_metric: Literal["recall_at_k", "mrr", "hit_rate_at_k", "balanced"] = "recall_at_k"
    min_floor_tokens: int = Field(200, ge=0)
    # balanced = recall_at_k - balanced_lambda * (retrieved_tokens / min_retrieved_tokens - 1)
    balanced_lambda: float = Field(0.05, ge=0.0)
    bootstrap_resamples: int = Field(10_000, ge=100)
    seed: int = 0


class StrategyConfig(BaseModel):
    name: str
    params: dict = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _known_strategy(cls, v: str) -> str:
        from chunklab.chunkers.registry import HIDDEN_STRATEGIES, available_strategies

        known = available_strategies()
        if v not in known and v not in HIDDEN_STRATEGIES:
            raise ValueError(f"unknown strategy '{v}'; available: {', '.join(sorted(known))}")
        return v


class OutputConfig(BaseModel):
    formats: list[Literal["console", "html", "json"]] = Field(
        default_factory=lambda: ["console", "html", "json"]
    )
    dir: str = "./chunklab_report"


class Config(BaseModel):
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    strategies: list[StrategyConfig] = Field(default_factory=list)
    output: OutputConfig = Field(default_factory=OutputConfig)


def default_strategies() -> list[StrategyConfig]:
    return [
        StrategyConfig(name="fixed", params={"chunk_size": 512, "overlap": 64}),
        StrategyConfig(name="recursive", params={"chunk_size": 512, "overlap": 64}),
        StrategyConfig(
            name="semantic",
            params={"breakpoint_percentile": 95, "min_tokens": 200, "max_tokens": 1000},
        ),
        StrategyConfig(
            name="semantic_no_floor",
            params={"breakpoint_percentile": 95, "min_tokens": 0, "max_tokens": 1000},
        ),
        StrategyConfig(name="structure", params={"max_tokens": 800}),
    ]


def default_config() -> Config:
    return Config(strategies=default_strategies())


def load_config(path: str | Path | None) -> Config:
    """Load a Config from YAML; with no path, return the default config."""
    if path is None:
        return default_config()
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    config = Config.model_validate(data)
    if not config.strategies:
        config.strategies = default_strategies()
    return config


def load_questions(path: str | Path) -> list[Question]:
    """Load questions from a YAML file with a top-level `questions:` list."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    raw = data.get("questions")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: expected a top-level 'questions:' list")
    return [Question.model_validate(q) for q in raw]
