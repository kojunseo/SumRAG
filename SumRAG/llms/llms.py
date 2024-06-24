import numpy as np
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_models.huggingface import ChatHuggingFace
# from .validator import OllamaValidation
from .property import AsIsProperty


class LLMs:
    """
    You can use the models by calling the static methods.

    Example:
        .. code-block:: python
        llms = LLMs.gpt3_5
        llms = LLMs.gpt4
        llms = LLMs.llama3_8b
    Possible models:
        gpt3_5, gpt4, llama3_8b, llama3_70b, llama3_kr, qwen2_kr
    """

    @AsIsProperty
    @staticmethod
    def gpt3_5() -> ChatOpenAI:
        """
        OpenAI's GPT-3.5 model with temperature 0
        """
        llms = ChatOpenAI(
            model='gpt-3.5-turbo',
            temperature=0,
        )
        return llms

    @AsIsProperty
    @staticmethod
    def gpt4() -> ChatOpenAI:
        llms = ChatOpenAI(
            model='gpt-4o',
            temperature=0,
        )
        return llms
    
    @AsIsProperty
    @staticmethod
    def llama3_8b() -> ChatOllama:
        # OllamaValidation("llama3:latest")
        llms = ChatOllama(
            model="llama3:latest"
        )
        return llms

    @AsIsProperty
    @staticmethod
    def llama3_70b() -> ChatOllama:
        # OllamaValidation("llama3:70b")
        llms = ChatOllama(
            model="llama3:70b"
        )
        return llms
    
    @AsIsProperty
    @staticmethod
    def llama3_kr() -> ChatOllama:
        # OllamaValidation("Llama-3-Open-Ko-8B-Q5_K_M:latest")
        llms = ChatOllama(
            model="Llama-3-Open-Ko-8B-Q5_K_M:latest"
        )
        return llms
    
    @AsIsProperty
    @staticmethod
    def qwen2_kr() -> ChatOllama:
        # OllamaValidation("qwen2-7b-instruct-q8_0:latest")
        llms = ChatOllama(
            model="qwen2-7b-instruct-q8_0:latest"
        )
        return llms

if __name__ == "__main__":
    # class property to a real value
    print(LLMs.gpt3_5)
    # print(LLMs.gpt4)