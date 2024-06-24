import os
import ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
# from .validator import OllamaValidation

from .property import AsIsProperty


class EMBs:
    """
    You can use the embedding by calling the static methods.

    Example:
        .. code-block:: python
        emb = EMBs.hf_kr
        emb = EMBs.openai
        emb = EMBs.llama3_8b
    Possible embeddings:
        hf_kr, openai, llama3_8b, llama3_70b, llama3_kr, qwen2_kr
    """

    @AsIsProperty
    @staticmethod
    def hf_kr():
        """
        HuggingFaceEmbeddings with Korean RoBERTa model.
        """
        emb = HuggingFaceEmbeddings(
            model_name='jhgan/ko-sroberta-nli',
            model_kwargs={'device':'cpu'},
            encode_kwargs={'normalize_embeddings':True},
        )
        return emb

    @AsIsProperty
    @staticmethod
    def openai():
        if os.getenv("OPENAI_API_KEY") is None:
            raise ValueError("Please set OPENAI_API_KEY in your environment variables following instruction https://wikidocs.net/233342")
        emb = OpenAIEmbeddings()
        return emb
    
    @AsIsProperty
    @staticmethod
    def llama3_8b():
        # OllamaValidation("llama3:latest")
        emb = OllamaEmbeddings(
            model="llama3:latest"
        )
        return emb
    
    @AsIsProperty
    @staticmethod
    def llama3_70b():
        OllamaEmbeddings("llama3:70b")
        emb = OllamaEmbeddings(
            model="llama3:70b"
        )
        return emb
    
    @AsIsProperty
    @staticmethod
    def llama3_kr():
        # OllamaValidation("Llama-3-Open-Ko-8B-Q5_K_M:latest")
        emb = OllamaEmbeddings(
            model="Llama-3-Open-Ko-8B-Q5_K_M:latest"
        )
        return emb

    @AsIsProperty
    @staticmethod
    def qwen2_kr():
        # OllamaValidation("qwen2-7b-instruct-q8_0:latest")
        emb = OllamaEmbeddings(
            model="qwen2-7b-instruct-q8_0:latest"
        )
        return emb
    