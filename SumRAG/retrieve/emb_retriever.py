from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy


class EMBRetriever:
    def __init__(self, emb, s_input):
        vectorstore = FAISS.from_documents(s_input.keyword_explains,
                                        embedding = emb,
                                        distance_strategy = DistanceStrategy.JACCARD)

        self.__vectorstore_retriever = vectorstore.as_retriever()
        self.__docs = s_input.docs

    def __call__(self, query):
        key = self.__vectorstore_retriever.invoke(query)[0].metadata["keyword"]
        return self.__docs[key]