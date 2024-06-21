from SumRAG.llms import get_llm
from SumRAG.embeddings import get_emb
from SumRAG.retrieve import HierRetriever, LLMRetriever
from SumRAG.generation import BasicGenerator
# from SumRAG.documents import load_from_files
from SumRAG.documents import SumInput
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks import get_openai_callback

from dotenv import load_dotenv
load_dotenv()

llm3 = get_llm("gpt3_5")
llm4 = get_llm("gpt4_0")
emb = get_emb("hf_kr")

# documents = SumInput.load_from_files("./ke_education_txt", llm3)
# documents.save("./ke_education_json")

documents = SumInput.load("./src/ke_education_json")

retriever = HierRetriever(llm=llm3, s_input=documents)
generator = BasicGenerator(llm=llm3, retriever_fn=retriever, output_parser=StrOutputParser())
print(generator("업무용 시설분담금의 환불은 어디에 물어봐야 하나요?"))