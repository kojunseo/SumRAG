from SumRAG import LLMs, EMBs
from SumRAG.retrieve import HierLLMRetriever, HierEMBMixRetriever, LLMRetriever, EMBRetriever
from SumRAG.generation import BasicGenerator, AdditionalQuestionGenerator
# from SumRAG.documents import load_from_files
from SumRAG.documents import SumInput
from langchain_community.callbacks import get_openai_callback

from dotenv import load_dotenv
load_dotenv()


# documents = SumInput.load_from_files("./ke_education_txt", LLMs.gpt3_5)
# documents.save("./ke_education_json")

documents = SumInput.load("./src/ke_education_json")

retriever = HierLLMRetriever(llm=LLMs.gpt3_5, s_input=documents)
# retriever = EMBRetriever(emb=EMBs.hf_kr, s_input=documents)
# retriever = HierLLMRetriever(emb=EMBs.hf_kr, llm=LLMs.gpt3_5, s_input=documents)

generator = BasicGenerator(llm=LLMs.gpt3_5, retriever_fn=retriever)
additional_generator = AdditionalQuestionGenerator(llm=LLMs.gpt3_5, retriever_fn=retriever)

question = "부취제가 무엇인가요?"

print(generator(question))

questions, answers = additional_generator(question)
print(questions)
print(answers)