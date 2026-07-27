"""On-disk embedding cache (roadmap 6.2).

Comparing five chunking strategies over one corpus re-embeds much of the same
text: strategies agree on most sentence boundaries, and a second run after
editing questions re-embeds the corpus unchanged. Embedding is the slowest step
by far, so vectors are memoized on disk.

The key covers the model *and* its resolved revision: a cache that survived a
model upgrade would silently serve vectors from the old weights, which is worse
than no cache. When the revision cannot be resolved the entry is still keyed by
model name, marked `unknown` — pinning less, but never mixing the two.

SQLite is used because it is in the standard library and gives us atomic
concurrent writes; a cache is not worth a dependency.
"""

import hashlib
import os
import sqlite3
from pathlib import Path

import numpy as np

from chunklab.embeddings.base import Embedder

#: Bump when the stored representation changes, to invalidate old entries.
CACHE_FORMAT = 1


def default_cache_path() -> Path:
    """Cache location, honoring CHUNKLAB_CACHE_DIR then XDG_CACHE_HOME."""
    override = os.environ.get("CHUNKLAB_CACHE_DIR")
    if override:
        return Path(override).expanduser() / "embeddings.sqlite3"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "chunklab" / "embeddings.sqlite3"


def caching_enabled() -> bool:
    return os.environ.get("CHUNKLAB_NO_CACHE", "").strip().lower() not in {"1", "true", "yes"}


class EmbeddingCache:
    """Content-addressed float32 vector store."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_cache_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path))
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vec BLOB)")
        self._db.commit()

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        # SQLite caps host parameters per statement, so read in blocks.
        for i in range(0, len(keys), 500):
            block = keys[i : i + 500]
            placeholders = ",".join("?" * len(block))
            rows = self._db.execute(
                f"SELECT key, vec FROM vectors WHERE key IN ({placeholders})", block
            )
            for key, blob in rows:
                found[key] = np.frombuffer(blob, dtype=np.float32)
        return found

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        rows = [
            (key, np.ascontiguousarray(vec, dtype=np.float32).tobytes())
            for key, vec in items.items()
        ]
        self._db.executemany("INSERT OR REPLACE INTO vectors (key, vec) VALUES (?, ?)", rows)
        self._db.commit()

    def close(self) -> None:
        self._db.close()


class CachedEmbedder:
    """Wraps an embedder, serving repeat texts from disk.

    Results are identical to the wrapped embedder: only the vectors it already
    produced for the same model revision are reused.
    """

    def __init__(self, inner: Embedder, cache: EmbeddingCache | None = None) -> None:
        self._inner = inner
        self._cache = cache if cache is not None else EmbeddingCache()
        self.model_name = inner.model_name
        self.max_seq_tokens = inner.max_seq_tokens
        self.revision = inner.revision
        self._signature = str(getattr(inner, "cache_signature", ""))
        self.hits = 0
        self.misses = 0

    def _key(self, text: str, role: str) -> str:
        # `role` keeps query and passage vectors apart: an asymmetric model gives
        # the same text two different embeddings depending on the side it is on.
        # `signature` covers anything else that changes the vector for identical
        # input text - today, whether instruction prefixes are applied.
        material = "\x00".join(
            (
                str(CACHE_FORMAT),
                self.model_name,
                self.revision or "unknown",
                self._signature,
                role,
                text,
            )
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "passage", self._inner.embed)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "query", self._inner.embed_queries)

    def _embed(self, texts: list[str], role: str, compute) -> np.ndarray:
        if not texts:
            return compute(texts)

        keys = [self._key(t, role) for t in texts]
        cached = self._cache.get_many(list(dict.fromkeys(keys)))

        # Dedupe within the batch too: the same text twice is one model call.
        pending: dict[str, str] = {}
        for key, text in zip(keys, texts, strict=True):
            if key not in cached and key not in pending:
                pending[key] = text

        self.hits += len(texts) - sum(1 for k in keys if k in pending)
        self.misses += len(pending)

        if pending:
            fresh = compute(list(pending.values()))
            computed = dict(zip(pending.keys(), fresh, strict=True))
            self._cache.put_many(computed)
            cached.update(computed)

        dims = {vec.shape[0] for vec in cached.values()}
        if len(dims) > 1:
            raise ValueError(
                f"embedding cache returned mixed dimensions {sorted(dims)} for "
                f"{self.model_name}; delete {self._cache.path} and re-run"
            )
        return np.stack([cached[key] for key in keys]).astype(np.float32)
