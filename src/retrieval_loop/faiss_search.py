from langchain.vectorstores import FAISS
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from langchain.docstore.base import AddableMixin, Docstore
from langchain.docstore.document import Document
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore
from langchain.vectorstores.utils import DistanceStrategy, maximal_marginal_relevance
import operator
import os
from langchain.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm



def dependable_faiss_import(no_avx2: Optional[bool] = None) -> Any:
    """
    Import faiss if available, otherwise raise error.
    If FAISS_NO_AVX2 environment variable is set, it will be considered
    to load FAISS with no AVX2 optimization.

    Args:
        no_avx2: Load FAISS strictly with no AVX2 optimization
            so that the vectorstore is portable and compatible with other devices.
    """
    if no_avx2 is None and "FAISS_NO_AVX2" in os.environ:
        no_avx2 = bool(os.getenv("FAISS_NO_AVX2"))

    try:
        if no_avx2:
            from faiss import swigfaiss as faiss
        else:
            import faiss
    except ImportError:
        raise ImportError(
            "Could not import faiss python package. "
            "Please install it with `pip install faiss-gpu` (for CUDA supported GPU) "
            "or `pip install faiss-cpu` (depending on Python version)."
        )
    return faiss

# define a new class write the similarity_search_with_score_by_vector method of FAISS class

class Batch_FAISS(FAISS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def similarity_search_with_score_by_vector(self,
            embeddings: List[List[float]],
            k: int = 4,
            filter: Optional[Dict[str, Any]] = None,
            fetch_k: int = 20,
            **kwargs: Any,
        ) -> List[List[Tuple[Document, float]]]:
        """Return docs most similar to query.

    Args:
        embeddings: List of embedding vectors to look up documents similar to.
        k: Number of Documents to return. Defaults to 4.
        filter (Optional[Dict[str, Any]]): Filter by metadata. Defaults to None.
        fetch_k: (Optional[int]) Number of Documents to fetch before filtering.
                  Defaults to 20.
        **kwargs: kwargs to be passed to similarity search. Can include:
            score_threshold: Optional, a floating point value between 0 to 1 to
                filter the resulting set of retrieved docs

    Returns:
        List of lists of documents most similar to each query text and L2 distance
        in float for each. Lower score represents more similarity.
    """
        faiss = dependable_faiss_import()
        vectors = np.array(embeddings, dtype=np.float32)
        if self._normalize_L2:
            faiss.normalize_L2(vectors)
        print(f'start searching, query size: {len(vectors)}')
        scores, indices = self.index.search(vectors, k if filter is None else fetch_k)
        results = []
        print(f'start fetching, query size: {len(indices)}')
        for idx, query_indices in tqdm(enumerate(indices), total=len(indices)):
            docs = []
            for j, i in enumerate(query_indices):
                if i == -1:
                    # This happens when not enough docs are returned.
                    continue
                _id = self.index_to_docstore_id[i]
                doc = self.docstore.search(_id)
                if not isinstance(doc, Document):
                    raise ValueError(f"Could not find document for id {_id}, got {doc}")
                if filter is not None:
                    filter = {
                        key: [value] if not isinstance(value, list) else value
                        for key, value in filter.items()
                    }
                    if all(doc.metadata.get(key) in value for key, value in filter.items()):
                        docs.append((doc, scores[idx][j]))
                else:
                    docs.append((doc, scores[idx][j]))

            score_threshold = kwargs.get("score_threshold")
            if score_threshold is not None:
                cmp = (
                    operator.ge
                    if self.distance_strategy
                       in (DistanceStrategy.MAX_INNER_PRODUCT, DistanceStrategy.JACCARD)
                    else operator.le
                )
                docs = [
                    (doc, similarity)
                    for doc, similarity in docs
                    if cmp(similarity, score_threshold)
                ]
            results.append(docs[:k])
        return results


class Batch_HuggingFaceEmbeddings(HuggingFaceEmbeddings):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def embed_queries(self, queries: List[str]) -> List[List[float]]:
        """Embed queries.

        Args:
            queries: List of queries to embed.

        Returns:
            List of embeddings.
        """
        return self.embed_documents(queries)
