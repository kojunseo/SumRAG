import os
import ollama
import requests
from tqdm import tqdm

class OllamaValidation:
    """
    Not Implemented yet.
    """
    @staticmethod
    def validate(model_name):
        # Check if model is installed
        ollama_model_name_list = [ol["name"] for ol in ollama.list()["models"]]
        if model_name not in ollama_model_name_list:
            OllamaValidation.install_model(model_name)
        return True
    

    @staticmethod
    def install_model(model_name):
        # Direct install from ollama server
        if model_name in ["llama3:latest", "llama3:70b"]:
            ollama.install(model_name)

        # get model from huggingface and install
        else:
            model_file = """FROM {filepath}
TEMPLATE \"""{{- if .System }}
<s>{{ .System }}</s>
{{- end }}
<s>Human:
{{ .Prompt }}</s>
<s>Assistant:
\"""

SYSTEM \"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\"""

PARAMETER temperature 0
PARAMETER num_predict 3000
PARAMETER num_ctx 4096
PARAMETER stop <s>
PARAMETER stop </s>
"""
            install_url = {
                "Llama-3-Open-Ko-8B-Q5_K_M:latest": "https://huggingface.co/teddylee777/Llama-3-Open-Ko-8B-gguf/resolve/main/Llama-3-Open-Ko-8B-Q5_K_M.gguf",
                "qwen2-7b-instruct-q8_0:latest": "https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF/resolve/main/qwen2-7b-instruct-q8_0.gguf",   
            }

            download_path = "./.ollama_gguf"
            if not os.path.exists(download_path):
                os.makedirs(download_path)
            
            gguf_path = os.path.join(download_path, model_name + ".gguf")
            if os.path.exists(gguf_path):
                print("Model already downloaded.. " + model_name, flush=True)
            else:
                print("Downloading model.. " + model_name, flush=True)
                with open(gguf_path, "wb") as f:
                    response = requests.get(install_url[model_name], stream=True)
                    total_size_in_bytes= int(response.headers.get('content-length', 0))
                    block_size = 1024
                    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        f.write(data)
                    progress_bar.close()

            model_file = model_file.format(filepath=gguf_path)
            model_file_path = os.path.join(download_path,"Modelfile")
            with open(model_file_path, "w") as f:
                f.write(model_file)
            
            os.system(f"ollama create {model_name} -f {model_file_path}")
        
        return True

            

if __name__ == "__main__":
    OllamaValidation.validate("qwen2-7b-instruct-q8_0:latest")