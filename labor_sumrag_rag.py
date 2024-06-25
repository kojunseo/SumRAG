# 사용법 관련 예제코드 -> 자체 개발한 SumRAG를 사용하여 문서 요약 및 질문 생성을 하는 코드입니다.

from SumRAG import LLMs, EMBs
from SumRAG.retrieve import HierLLMRetriever, HierEMBMixRetriever, EMBRetriever, LLMRetriever
from SumRAG.generation import BasicGenerator
from SumRAG.generation import AdditionalQuestionGenerator
from SumRAG.documents import SumInput
from dotenv import load_dotenv

load_dotenv()


documents = SumInput.load("./example/documents")

# retriever = HierLLMRetriever(llm=LLMs.gpt3_5, s_input=documents)
retriever = HierLLMRetriever( llm=LLMs.gpt3_5, s_input=documents) 
generator = BasicGenerator(llm=LLMs.gpt3_5, retriever_fn=retriever)

question = "80% 미만으로 근무한 근로자에게 며칠의 유급휴가를 주어야 할까?"

answer = generator(question)
print("===" * 20)
print(f"[HUMAN]\n{question}\n") # 80% 미만으로 근무한 근로자에게 며칠의 유급휴가를 주어야 할까?
print(f"[AI]\n{answer}") # 80% 미만으로 근무한 근로자에게는 한 달에 한 번씩 근무한 개월 수에 따라 유급휴가를 주어야 합니다. (정답)


# 추가질문 생성 -> 아래 기능은 아직은 실험적인 기능이기 때문에, 오류가 발생하거나 결과가 나오지 않을 수 있습니다.
# additional_generator = AdditionalQuestionGenerator(llm=LLMs.gpt3_5, retriever_fn=retriever)
# add_q = additional_generator(question)

# print("추가질문 1: ", add_q[0][0])
# print("답변: ", add_q[1][0])
# print()
# print("추가질문 2: ", add_q[0][1])
# print("답변: ", add_q[1][1])