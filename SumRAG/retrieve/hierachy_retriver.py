from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.documents import Document


class HierRetriever:
    def __init__(self, llm, s_input):
        self.__llm = llm
        self.__docs = s_input.docs

        prompt1 = ChatPromptTemplate.from_template('''Following is the explanation of the document titles of {keywords}:
        {context}

        Return a single keyword between {keywords} that is most relevant to the user's question.
        Question: {question}
        Just return the keyword, not the explanation.
        ''')
        prompt2 = ChatPromptTemplate.from_template("""Following is the explanation of the each document titles of {length} documents:
        {context}
                                                   
        What are the necessary documents for answering the user's question? Answer with the index of the documents separated by space, and no need to include the explanation.
        User's question: {question}
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
        # keyword_explains = [Document(page_content=dc.metadata["keyword"]) for dc in doc_from_1]
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


        