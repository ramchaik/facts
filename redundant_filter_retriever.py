from typing import Coroutine, Dict, List
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import BaseRetriever
from langchain_core.documents import Document

"""
This retriever will filter out duplicate embeddings that can be fond in vectorstore before being feed to LLM.
Goal is to produce correct results even if the vectorstore has redundant data (text and embeddings)

1. calculates the embedding score for the input query.
2. uses the embeddings and compares to the embeddings in vectorstore (using max_marginal_relevance_search_by_vector method of chroma)
3. removes duplicates embeddings found in the vector score (lambda_mult: 0-1, higher values allows for duplication of embeddings)

"""
class RedundantFilterRetriever(BaseRetriever):
    embeddings: Embeddings
    chroma: Chroma

    def get_relevant_documents(self, query) -> List[Document]:
        emb = self.embeddings.embed_query(query)

        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=emb,
            lambda_mult=0.8
        )

    def aget_relevant_documents():
        return []