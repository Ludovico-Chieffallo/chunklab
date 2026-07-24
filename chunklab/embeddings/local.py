"""Local embedder via sentence-transformers (default: BAAI/bge-small-en-v1.5)."""

import numpy as np

from chunklab.embeddings.base import normalize


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
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.max_seq_tokens: int | None = getattr(self._model, "max_seq_length", None)
        self.revision: str | None = _resolve_cached_revision(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True
        )
        return normalize(np.asarray(vectors))
