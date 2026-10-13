"""Config schema, YAML loading, and defaults."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

from chunklab.errors import (
    InputError,
    InputNotFoundError,
    describe_validation_error,
    describe_yaml_error,
)
from chunklab.models import Question


class EmbeddingConfig(BaseModel):
    backend: Literal["local", "openai", "fake"] = "local"
    model: str = "BAAI/bge-small-en-v1.5"
    # Reuse vectors across runs and strategies. Keyed by model revision, so it
    # never serves vectors from different weights. CHUNKLAB_NO_CACHE=1 also wins.
    cache: bool = True
    # Apply the query/passage instruction prefixes the model was trained with.
    # Mandatory for E5; set false only to reproduce a run made without them.
    prefixes: bool = True

    @field_validator("backend")
    @classmethod
    def _implemented(cls, v: str) -> str:
        # Part of the schema so a config written for it still parses, refused here
        # so it fails with the config instead of after every document has loaded.
        if v == "openai":
            raise ValueError("'openai' is not available yet; use 'local'")
        return v


class RetrievalConfig(BaseModel):
    mode: Literal["dense", "bm25", "hybrid"] = "dense"
    top_k: int = Field(5, ge=1)
    # When set, every strategy is evaluated under every listed retriever and the
    # report becomes a strategy x retriever matrix. Empty means `mode` alone.
    compare: list[Literal["dense", "bm25", "hybrid"]] = Field(default_factory=list)

    @field_validator("compare")
    @classmethod
    def _no_duplicates(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError(f"retrieval.compare lists a mode twice: {v}")
        return v

    @property
    def modes(self) -> list[str]:
        return list(self.compare) if self.compare else [self.mode]


class EvalConfig(BaseModel):
    fuzzy_threshold: float = Field(0.90, ge=0.0, le=1.0)
    ranking_metric: Literal["balanced", "recall_at_k", "mrr", "hit_rate_at_k"] = "balanced"
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


def _read_yaml(path: str | Path, what: str):
    """Parse a YAML file the user named, turning each way it can fail into an
    InputError that says which file and where."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise InputNotFoundError(f"{what} not found: {path}") from None
    except UnicodeDecodeError:
        raise InputError(f"{what} is not UTF-8 text: {path}") from None
    except OSError as exc:
        raise InputError(f"cannot read {what} {path}: {exc.strerror or exc}") from None
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise InputError(f"{path} is not valid YAML: {describe_yaml_error(exc)}") from None


def load_config(path: str | Path | None) -> Config:
    """Load a Config from YAML; with no path, return the default config."""
    if path is None:
        return default_config()
    data = _read_yaml(path, "config file") or {}
    try:
        config = Config.model_validate(data)
    except ValidationError as exc:
        raise InputError(f"{path}: {describe_validation_error(exc)}") from None
    if not config.strategies:
        config.strategies = default_strategies()
    return config


def load_questions(path: str | Path) -> list[Question]:
    """Load questions from a YAML file with a top-level `questions:` list."""
    data = _read_yaml(path, "questions file")
    raw = data.get("questions") if isinstance(data, dict) else None
    if not isinstance(raw, list) or not raw:
        raise InputError(f"{path}: expected a top-level 'questions:' list")
    questions = []
    for number, item in enumerate(raw, 1):
        try:
            questions.append(Question.model_validate(item))
        except ValidationError as exc:
            ident = item.get("id") if isinstance(item, dict) else None
            label = f"question {number}" + (f" ({ident})" if ident else "")
            raise InputError(f"{path}: {label}: {describe_validation_error(exc)}") from None
    return questions
