from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain.prompts import ChatPromptTemplate


class GeneratorCore:
    def __init__(self, retriever_fn, llm, template, output_parser=StrOutputParser()):
        
        prompt = ChatPromptTemplate.from_template(template)
        self.chain = (
            {'context': itemgetter("question") | RunnableLambda(retriever_fn), 'question': itemgetter("question")}
            | prompt
            | llm
            | output_parser
        )

    def __call__(self, query):
        return self.chain.invoke({"question":query})