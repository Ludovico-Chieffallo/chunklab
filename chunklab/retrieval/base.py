"""Retriever protocol."""

from typing import Protocol, runtime_checkable

from chunklab.models import RetrievedChunk


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]: ...
