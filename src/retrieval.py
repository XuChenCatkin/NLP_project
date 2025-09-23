# src/retrieval.py
import faiss
import numpy as np
from abc import ABC, abstractmethod
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from typing import Union
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Base Class ---
class BaseRetriever(ABC):
    """Abstract Base Class for all retriever models."""
    def __init__(self, passages: list[str], chunk_ids: list[str]):
        self.passages = passages
        self.chunk_ids = chunk_ids
        self.index = self._build_index()

    @abstractmethod
    def _build_index(self):
        """Builds the search index from the passages."""
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[dict]:
        """Retrieves the top_k most relevant documents for a given query."""
        pass
    
    def _format_results(self, indices: list[int], scores: list[float], query: str = None) -> list[dict]:
        """Formats raw results into a structured list of dictionaries."""
        results = []
        for i, score in zip(indices, scores):
            result = {
                "chunk_id": self.chunk_ids[i],
                "passage": self.passages[i],
                "score": float(score)
            }
            if query:
                result["sub_query"] = query
            results.append(result)
        return results

# --- Sparse Retrievers ---
class TfidfRetriever(BaseRetriever):
    """Retrieves documents using TF-IDF."""
    def _build_index(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', norm='l2')
        return self.vectorizer.fit_transform(self.passages)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        query_vec = self.vectorizer.transform([query])
        scores = (self.index @ query_vec.T).toarray().flatten()
        sorted_indices = np.argsort(scores)[::-1][:top_k]
        return self._format_results(sorted_indices, scores[sorted_indices])

class BM25Retriever(BaseRetriever):
    """Retrieves documents using BM25."""
    def _build_index(self):
        tokenized_passages = [word_tokenize(p.lower()) for p in self.passages]
        return BM25Okapi(tokenized_passages)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        tokenized_query = word_tokenize(query.lower())
        scores = self.index.get_scores(tokenized_query)
        sorted_indices = np.argsort(scores)[::-1][:top_k]
        return self._format_results(sorted_indices, scores[sorted_indices])

# --- Dense Retriever for Pre-computed Queries ---
class PrecomputedDenseRetriever:
    """
    Handles dense retrieval where query embeddings are pre-computed and stored in a Faiss index.
    """
    def __init__(self, corpus_data: list[dict], corpus_index_path: str, 
                 query_list: list[str], query_index_path: str):
        self.corpus_data = corpus_data
        self.corpus_index = faiss.read_index(corpus_index_path)
        self.query_list = query_list
        self.query_index = faiss.read_index(query_index_path)
        print("PrecomputedDenseRetriever initialized.")

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Finds the pre-computed embedding for a given query string."""
        try:
            position = self.query_list.index(query)
            return self.query_index.reconstruct(position)
        except ValueError:
            raise ValueError(f"Query '{query}' not found in the pre-computed query list.")

    def retrieve(self, queries: Union[list[str], str], top_k: int = 5, return_full_doc: bool = False) -> list[dict]:
        """
        Retrieves documents for one or more queries.

        Args:
            queries: A single query string or a list of query strings.
            top_k: The number of documents to return for each query.
            return_full_doc: If True, returns the entire original document dict from the corpus.
                             If False, returns a formatted result with chunk_id, passage, and score.
        """
        if isinstance(queries, str):
            queries = [queries]

        results = []
        for query in queries:
            query_emb = self._get_query_embedding(query)
            query_emb = query_emb.reshape(1, -1)  # Reshape to (1, d)
            
            distances, indices = self.corpus_index.search(query_emb, top_k)
            
            if return_full_doc:
                results.extend([self.corpus_data[i] for i in indices[0]])
            else:
                # Format results similar to other retrievers
                formatted = [
                    {
                        "sub_query": query,
                        "chunk_id": self.corpus_data[i]["chunk_id"],
                        "passage": self.corpus_data[i]["passage"],
                        "score": float(score) # Lower distance is better in Faiss L2
                    } for i, score in zip(indices[0], distances[0])
                ]
                results.extend(formatted)
        return results