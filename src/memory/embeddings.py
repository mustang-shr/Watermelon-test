"""
Embedding provider for instruction similarity (used to populate SIMILAR_TO edges
and to drive the duplicate-issue-detection synthesized tool).

Two implementations, one interface:

- LocalTFIDFEmbedder: no network call, works fully offline. This is what's used
  in CI/tests and in this sandbox where build.nvidia.com isn't reachable. It is
  NOT semantically strong - it's lexical (shared-words) similarity, not meaning
  similarity. Good enough for surfacing "you've basically asked this before"
  on near-duplicate phrasing, not good enough as the real duplicate-issue-detector.

- NvidiaNIMEmbedder: calls a NIM embedding model (e.g. nvidia/nv-embedqa-e5-v5).
  This is the one you actually run for the demo - real semantic similarity is the
  point of the duplicate-issue-detection capability-synthesis example. Requires
  NVIDIA_API_KEY and cannot be exercised in this sandbox (build.nvidia.com is not
  on the allowed egress list here) - test this one locally.

Swap is a single line in config.py, not a rewrite.
"""

from abc import ABC, abstractmethod
import numpy as np
import warnings


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n_texts, dim) array. Dim need not match across providers."""
        raise NotImplementedError

    def similarity(self, text_a: str, text_b: str) -> float:
        vecs = self.embed([text_a, text_b])
        a, b = vecs[0], vecs[1]
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)


class LocalTFIDFEmbedder(EmbeddingProvider):
    """Offline fallback. Fit-on-the-fly TF-IDF since we don't have a fixed corpus."""

    def embed(self, texts: list[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(texts)
        return matrix.toarray()


class NvidiaNIMEmbedder(EmbeddingProvider):
    """Production embedder via NVIDIA NIM.

    Model: nvidia/nv-embedqa-e5-v5 — verified active on integrate.api.nvidia.com
    as of June 2026. Uses input_type="query" as required by the NeMo Retriever
    embedding API. Returns 1024-dim float32 vectors.
    """

    def __init__(self, api_key: str, model: str = "nvidia/nv-embedqa-e5-v5",
                 base_url: str = "https://integrate.api.nvidia.com/v1"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    def embed(self, texts: list[str]) -> np.ndarray:
        import requests
        try:
            resp = requests.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "input": texts,
                    "model": self.model,
                    "input_type": "query",
                },
                timeout=30,
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"NIM embeddings request timed out after 30s (model='{self.model}'). Run "
                f"`python scripts/check_nim_connection.py` to check connectivity separately."
            ) from None
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"Could not connect to NIM embeddings endpoint: {e}") from None
        resp.raise_for_status()
        data = resp.json()["data"]
        return np.array([d["embedding"] for d in data])


class FallbackEmbedder(EmbeddingProvider):
    """Try a semantic provider first, then fall back to a deterministic local one.

    This keeps the live demo usable when the remote embedding endpoint is down,
    misconfigured, or returns a 404 for the current model. In production the agent
    ran a real issue-creation task successfully while the NIM embedding endpoint
    was 404-ing — the FallbackEmbedder is why it didn't crash, it just degraded
    gracefully to TF-IDF similarity instead of semantic embeddings.

    Confidence note: if the primary is actually failing, plan reuse based on
    semantic similarity becomes less accurate (TF-IDF only catches near-identical
    phrasing, not genuine paraphrase). The WARNING printed is intentional — this
    degradation should be visible, not silently invisible.
    """

    def __init__(self, primary: EmbeddingProvider, fallback: EmbeddingProvider):
        self.primary = primary
        self.fallback = fallback

    def embed(self, texts: list[str]) -> np.ndarray:
        try:
            return self.primary.embed(texts)
        except Exception as exc:
            warnings.warn(
                f"Primary embedding provider failed ({exc.__class__.__name__}: {exc}); "
                f"falling back to local embeddings.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self.fallback.embed(texts)
