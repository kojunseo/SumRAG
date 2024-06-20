from SumRAG.llms import get_llm
from SumRAG.embeddings import get_emb
from SumRAG.retrieve import LLMRetriever, EMBRetriever
from SumRAG.generation import BasicGenerator
from SumRAG.documents import load_from_files
from SumRAG.documents import SumInput
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

llm = get_llm("gpt3_5")
emb = get_emb("hf_kr")

documents = SumInput.load("/home/raondata/kojunseo/RAG/SumRAG/ke_education_json")

retriever = EMBRetriever(emb=emb, s_input=documents)
generator = BasicGenerator(llm=llm, retriever_fn=retriever, output_parser=StrOutputParser())
print(generator("LPG의 생산방법은 무엇인가요?"))