from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough


class LLMRetriever:
    def __init__(self, llm, s_input):
        self.__llm = llm
        self.__docs = s_input.docs


        template = '''You are keyword based retreiver. Following context is the explanation of the document titles:
        {context}

        Return a single keyword between {keywords} that is most relevant to the user's question. Only return keyword string, not the explanation or punctuations.
        Question: {question}
        Answer: 
        '''

        prompt = ChatPromptTemplate.from_template(template)
        
        self.retrieve_chain = (
            {'context': RunnablePassthrough(), 'question': RunnablePassthrough(), 'keywords': RunnablePassthrough()}
            | prompt
            | self.__llm
            | StrOutputParser()
        )

        self.__keyword_explains = s_input.keyword_explains
        self.__keywords = s_input.keywords


    
    def __call__(self, query):
        key = self.retrieve_chain.invoke(
            {
                "context": self.__keyword_explains,
                "question": query,
                "keywords": self.__keywords
            }
        )
        return self.__docs[key]