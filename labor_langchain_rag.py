# 사용법 관련 예제코드 -> 일반적인 langchain을 사용한 예제코드와 비교하여 어떻게 사용하는지 확인

from langchain import hub
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


load_dotenv()

loader = TextLoader("./example/labor_standards_act.txt")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

splits = text_splitter.split_documents(docs)

vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt")

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)


# 단계 7: 체인 생성(Create Chain)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

question = "80% 미만으로 근무한 근로자에게 며칠의 유급휴가를 주어야 할까?"
response = rag_chain.invoke(question)

# 결과 출력
print("===" * 20)
print(f"[HUMAN]\n{question}\n") #80% 미만으로 근무한 근로자에게 며칠의 유급휴가를 주어야 할까?
print(f"[AI]\n{response}") # 80% 미만으로 근무한 근로자에게는 15일의 유급휴가를 주어야 합니다. 15일의 유급휴가는 1년 동안 80% 이상 근무한 근로자에게 주어져야 합니다. 15일의 유급휴가는 1년 미만으로 근무한 근로자에게도 주어져야 합니다.

# 결과 오류 -> 근로기준법상 80%미만으로 근무한 근로자에게는 한달에 1번씩 근무한 날 수에 따라 지급해야함.