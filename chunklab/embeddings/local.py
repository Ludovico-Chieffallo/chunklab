"""Local embedder via sentence-transformers (default: BAAI/bge-small-en-v1.5)."""

import numpy as np

from chunklab.embeddings.base import normalize
from chunklab.embeddings.prefixes import PrefixScheme, scheme_for


def _resolve_cached_revision(model_name: str) -> str | None:
    """Commit hash of the locally cached HF snapshot for `model_name`, if unambiguous."""
    try:
        from huggingface_hub import scan_cache_dir

        for repo in scan_cache_dir().repos:
            if repo.repo_id == model_name and repo.repo_type == "model":
                revisions = list(repo.revisions)
                if len(revisions) == 1:
                    return revisions[0].commit_hash
                for rev in revisions:
                    if "main" in rev.refs:
                        return rev.commit_hash
        return None
    except Exception:
        return None


class LocalEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", prefixes: bool = True) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.max_seq_tokens: int | None = getattr(self._model, "max_seq_length", None)
        self.revision: str | None = _resolve_cached_revision(model_name)
        # The scheme the model was *trained* with; applying it is what makes an
        # asymmetric model work at all (E5) or work better (BGE).
        self.prefixes: PrefixScheme = scheme_for(model_name) if prefixes else PrefixScheme()

    @property
    def cache_signature(self) -> str:
        """Anything besides model+revision that changes the vector for a given text."""
        return f"q={self.prefixes.query}|p={self.prefixes.passage}"

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True
        )
        return normalize(np.asarray(vectors))

    def embed(self, texts: list[str]) -> np.ndarray:
        prefix = self.prefixes.passage
        return self._encode([prefix + t for t in texts] if prefix else texts)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        prefix = self.prefixes.query
        return self._encode([prefix + t for t in texts] if prefix else texts)
