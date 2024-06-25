from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy


class HierEMBMixRetriever:
    """
    Hierarchical retriever that retrieves documents hierarchically with the help of embeddings and language model.
    1. Retrieve the document titles that are most relevant to the user's question with the help of embeddings.
    2. Inside the retrieved documents, retrieve the necessary documents for answering the user's question hierarchically with the help of language model.
    """
    def __init__(self, emb, llm, s_input):
        """
        Parameters:
        - emb: Embedding that is used for retrieving documents
        - llm: Language model that is used for retrieving documents
        - s_input: SumInput object that contains documents and necessary information for retrieving

        Example:
        ```python
        from SumRAG import EMBs, LLMs
        from SumRAG.retrieve import HierEMBMixRetriever
        from SumRAG.documents import SumInput

        documents = SumInput.load("./output_json")
        retriever = HierEMBMixRetriever(emb=EMBs.hf_kr, llm=LLMs.gpt3_5, s_input=documents)
        ```

        """
        self.__emb = emb
        self.__llm = llm
        self.__docs = s_input.docs

        self.__vectorstore_retriever = FAISS.from_documents(s_input.keyword_explains,
                                                            embedding = emb,
                                                            distance_strategy = DistanceStrategy.JACCARD).as_retriever()

        prompt = ChatPromptTemplate.from_template("""Following is the explanation of the each document titles of {length} documents:
        {context}
                                                   
        User's question: {question}
        What are the necessary documents for answering the user's question? Answer with the index of the documents separated by space, and no need to include the explanation. You can choose a single document or multiple documents.
        """)
        
        self.retrieve_chain = (
            {'context': RunnablePassthrough(), 'question': RunnablePassthrough(), 'length': RunnablePassthrough()}
            | prompt
            | self.__llm
            | StrOutputParser()
        )

        self.__keyword_explains = s_input.keyword_explains
        self.__keywords = s_input.keywords

    def __call__(self, query):
        key = self.__vectorstore_retriever.invoke(query)[0].metadata["keyword"]
        doc_from_1 = self.__docs[key]
        keyword_explains = "\n".join([f"{idx}: " +dc.metadata["keyword"] for idx, dc in enumerate(doc_from_1)])
        doc_from_2 = self.retrieve_chain.invoke(
            {
                "context": keyword_explains,
                "question": query,
                "length": len(doc_from_1)
            }
        )
        list_doc = map(int, doc_from_2.split(" "))
        
        out = [doc_from_1[i] for i in list_doc]
        return out


class HierLLMRetriever:
    """
    Hierarchical retriever that retrieves documents hierarchically with the help of language model.
    1. Retrieve the document titles that are most relevant to the user's question.
    2. Inside the retrieved documents, retrieve the necessary documents for answering the user's question hierarchically.
    """
    def __init__(self, llm, s_input):
        """
        Parameters:
        - llm: Language model that is used for retrieving documents
        - s_input: SumInput object that contains documents and necessary information for retrieving

        Example:
        ```python
        from SumRAG import LLMs
        from SumRAG.retrieve import HierRetriever
        from SumRAG.documents import SumInput

        documents = SumInput.load("./output_json")
        retriever = HierRetriever(llm=LLMs.gpt3_5, s_input=documents)
        ```

        """
        self.__llm = llm
        self.__docs = s_input.docs

        prompt1 = ChatPromptTemplate.from_template('''You are keyword based retreiver. Following context is the explanation of the document titles:
        {context}

        Return a single keyword between {keywords} that is most relevant to the user's question. Just return the keyword, not the explanation.
        Question: {question}
        Answer: 
        ''')
        prompt2 = ChatPromptTemplate.from_template("""Following is the explanation of the each document titles of {length} documents:
        {context}
                                                   
        User's question: {question}
        What are the necessary documents for answering the user's question? Answer with the index of the documents separated by space, and no need to include the explanation. You can choose a single document or multiple documents.
        """)
        
        self.retrieve_chain1 = (
            {'context': RunnablePassthrough(), 'question': RunnablePassthrough(), 'keywords': RunnablePassthrough()}
            | prompt1
            | self.__llm
            | StrOutputParser()
        )

        self.retrieve_chain2 = (
            {'context': RunnablePassthrough(), 'question': RunnablePassthrough(), 'keywords': RunnablePassthrough(), 'length': RunnablePassthrough()}
            | prompt2
            | self.__llm
            | StrOutputParser()
        )

        self.__keyword_explains = s_input.keyword_explains
        self.__keywords = s_input.keywords


    
    def __call__(self, query):
        key = self.retrieve_chain1.invoke(
            {
                "context": self.__keyword_explains,
                "question": query,
                "keywords": self.__keywords
            }
        )
        doc_from_1 = self.__docs[key]
        keyword_explains = "\n".join([f"{idx}: " +dc.metadata["keyword"] for idx, dc in enumerate(doc_from_1)])
        keywords = [dc.metadata["keyword"] for dc in doc_from_1]
        doc_from_2 = self.retrieve_chain2.invoke(
            {
                "context": keyword_explains,
                "question": query,
                "keywords": keywords,
                "length": len(doc_from_1)
            }
        )
        list_doc = map(int, doc_from_2.split(" "))
        
        out = [doc_from_1[i] for i in list_doc]
        return out


        