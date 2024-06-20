from difflib import get_close_matches
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

possible_embs = ["hf_kr", "openai", "llama3:8b", "llama3:70b", "llama3:kr", "qwen2:kr"]

def get_emb(emb_name):
    if emb_name == "hf_kr":

        emb_model_hf = HuggingFaceEmbeddings(
            model_name='jhgan/ko-sroberta-nli',
            model_kwargs={'device':'cpu'},
            encode_kwargs={'normalize_embeddings':True},
        )
        return emb_model_hf
    elif emb_name == "openai":
        emb_model_openai = OpenAIEmbeddings()
        return emb_model_openai
    elif emb_name == "llama3:8b":
        emb_model_llama3_8b = OllamaEmbeddings(
            model="llama3:latest"
        )
        return emb_model_llama3_8b
    elif emb_name == "llama3:70b":
        emb_model_llama3_70b = OllamaEmbeddings(
            model="llama3:70b"
        )
        return emb_model_llama3_70b
    elif emb_name == "llama3:kr":
        emb_model_llama3_kr = OllamaEmbeddings(
            model="Llama-3-Open-Ko-8B-Q5_K_M:latest"
        )
        return emb_model_llama3_kr
    elif emb_name == "qwen2:kr":
        emb_model_qwen2_kr = OllamaEmbeddings(
            model="qwen2-7b-instruct-q8_0:latest"
        )
        return emb_model_qwen2_kr


    else:
        is_this = get_close_matches(emb_name, possible_embs, n=1)[0]
        raise ValueError(f"Unknown emb name: '{emb_name}'. Did you mean '{is_this}'?")