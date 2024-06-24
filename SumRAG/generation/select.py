from .core import GeneratorCore
from langchain_core.output_parsers import StrOutputParser

class SelectNGenerator(GeneratorCore):
    def __init__(self, retriever_fn, llm, output_parser=StrOutputParser()):
        
        template = """Answer the question based only on the following context:
                {context}
                You don't need to use all the context, just use the necessary information.

                Question: {question}
            """
        super().__init__(retriever_fn, llm, template, output_parser)