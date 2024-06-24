from .core import GeneratorCore
from langchain_core.output_parsers import StrOutputParser

class BasicGenerator(GeneratorCore):
    def __init__(self, retriever_fn, llm, output_parser=StrOutputParser()):
        
        template = """Answer the question based only on the following context:
                {context}

                Question: {question}
            """
        super().__init__(retriever_fn, llm, template, output_parser)