from typing import List, Dict, Any
import numpy as np

from semantic_image_search.src.vectorstore.faiss_store import InMemoryVectorStore


def build_store_from_arrays(ids: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]] = None) -> InMemoryVectorStore:
	if metadatas is None:
		metadatas = [{} for _ in ids]
	store = InMemoryVectorStore()
	items = []
	for i, id_ in enumerate(ids):
		items.append({"id": id_, "embedding": embeddings[i], "metadata": metadatas[i]})
	store.add_many(items)
	return store


def retrieve(store: InMemoryVectorStore, query_embedding: List[float], k: int = 10) -> List[Dict[str, Any]]:
	"""Retrieve top-k nearest items from given store for the query embedding."""
	return store.retrieve(query_embedding, k=k)


__all__ = ["build_store_from_arrays", "retrieve"]
