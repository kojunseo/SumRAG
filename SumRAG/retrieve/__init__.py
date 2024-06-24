"""
Retriever of SumRAG is a module that search and filtering documents that are relevant to the user's question.
You can also make a new retriever with `__call__` method that takes query as an argument and returns the relevant documents.

Possible retrievers are:
- EMBRetriever: Retrieve documents with the help of embeddings
- LLMRetriever: Retrieve documents with the help of language model
- HierRetriever: Retrieve documents hierarchically with the help of language model
"""

from .emb_retriever import EMBRetriever
from .llm_retriever import LLMRetriever
from .hierachy_retriver import HierLLMRetriever, HierEMBMixRetriever