import numpy as np
from typing import List, Dict, Any, Optional


class InMemoryVectorStore:
	"""A tiny in-memory vector store that supports adding vectors and
	retrieving nearest neighbors by cosine similarity. This is a simple
	replacement for a FAISS-backed store used for tests and local runs.
	"""

	def __init__(self, dim: Optional[int] = None) -> None:
		self.dim = dim
		self.ids: List[str] = []
		self.embeddings: Optional[np.ndarray] = None
		self.metadatas: List[Dict[str, Any]] = []

	def add(self, id: str, embedding: List[float], metadata: Dict[str, Any] = None) -> None:
		emb = np.asarray(embedding, dtype=np.float32)
		if self.dim is None:
			self.dim = emb.shape[0]
		if emb.shape[0] != self.dim:
			raise ValueError("Embedding dimensionality mismatch")
		if self.embeddings is None:
			self.embeddings = emb.reshape(1, -1)
		else:
			self.embeddings = np.vstack([self.embeddings, emb.reshape(1, -1)])
		self.ids.append(id)
		self.metadatas.append(metadata or {})

	def add_many(self, items: List[Dict[str, Any]]) -> None:
		"""items: list of {'id': str, 'embedding': list/np.array, 'metadata': dict}
		"""
		for it in items:
			self.add(it["id"], it["embedding"], it.get("metadata"))

	def _ensure_store(self) -> None:
		if self.embeddings is None or len(self.ids) == 0:
			raise RuntimeError("No vectors in store")

	def retrieve(self, query_embedding: List[float], k: int = 10) -> List[Dict[str, Any]]:
		"""Return top-k nearest items by cosine similarity.

		Returns a list of dicts with keys: `id`, `score`, `metadata`.
		"""
		self._ensure_store()
		q = np.asarray(query_embedding, dtype=np.float32)
		if q.shape[0] != self.dim:
			raise ValueError("Query embedding dimensionality mismatch")

		# cosine similarity: (q . x) / (||q|| * ||x||)
		x = self.embeddings  # (N, D)
		q_norm = np.linalg.norm(q)
		x_norm = np.linalg.norm(x, axis=1)
		# avoid division by zero
		denom = (q_norm * x_norm) + 1e-12
		sims = (x @ q) / denom

		# get top-k indices
		k = min(k, sims.shape[0])
		top_idx = np.argsort(-sims)[:k]
		results = []
		for idx in top_idx:
			results.append({
				"id": self.ids[idx],
				"score": float(sims[idx]),
				"metadata": self.metadatas[idx],
			})
		return results


__all__ = ["InMemoryVectorStore"]
