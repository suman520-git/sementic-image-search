import numpy as np

from semantic_image_search.src.services.vectorstore_service import build_store_from_arrays, retrieve


def test_simple_retrieval_top1():
	ids = ["a", "b", "c"]
	embeddings = [
		[1.0, 0.0, 0.0],  # a
		[0.0, 1.0, 0.0],  # b
		[0.0, 0.0, 1.0],  # c
	]
	store = build_store_from_arrays(ids, embeddings)

	# query similar to "b"
	q = [0.0, 0.9, 0.1]
	results = retrieve(store, q, k=1)
	assert len(results) == 1
	assert results[0]["id"] == "b"


def test_retrieval_ranking():
	ids = ["x", "y"]
	embeddings = [[1.0, 0.0], [0.9, 0.1]]
	store = build_store_from_arrays(ids, embeddings)
	q = [1.0, 0.0]
	results = retrieve(store, q, k=2)
	assert results[0]["id"] == "x"
	assert results[1]["id"] == "y"
