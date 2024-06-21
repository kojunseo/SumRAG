from difflib import get_close_matches
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
# from langchain.llms import HuggingFaceHub
from langchain_community.chat_models.huggingface import ChatHuggingFace


possible_llms = ["hf_kr", "llama3_8b", "llama3_70b", "llama3_kr", "qwen2_kr", "ggml_kr", "gpt3_5", "gpt4_0"]
def get_llm(llm_name):
    if llm_name in ["gpt3.5" , "gpt3_5", "chatgpt3_5", "chatgpt3.5"]:
        llms = ChatOpenAI(
            model='gpt-3.5-turbo',
            temperature=0,
        )
    elif llm_name in ["gpt4.0", "gpt4_0", "chatgpt4_0", "chatgpt4.0"]:
        llms = ChatOpenAI(
            model='gpt-4o',
            temperature=0,
        )
    
    elif llm_name in ["llama3_8b"]:
        llms = ChatOllama(
            model="llama3:latest"
        )
    
    elif llm_name in ["llama3_70b"]:
        llms = ChatOllama(
            model="llama3:70b"
        )
    
    elif llm_name in ["llama3_kr"]:
        llms = ChatOllama(
            model="Llama-3-Open-Ko-8B-Q5_K_M:latest"
        )
    
    elif llm_name in ["qwen2_kr"]:
        llms = ChatOllama(
            model="qwen2-7b-instruct-q8_0:latest"
        )
    
    elif llm_name in ["ggml_kr"]:
        llms = ChatOllama(
            model="ggml-model-Q5_K_M:latest"
        )
    
    else:
        # similar name
        is_this = get_close_matches(llm_name, possible_llms, n=1)[0]
        raise ValueError(f"Unknown llm name: '{llm_name}'. Did you mean '{is_this}'?")
    
    return llms

