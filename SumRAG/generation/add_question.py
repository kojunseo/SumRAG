from .core import GeneratorCore
from langchain_core.output_parsers import StrOutputParser

class AdditionalQuestionGenerator(GeneratorCore):
    def __init__(self, retriever_fn, llm, output_parser=StrOutputParser()):
        
        template = """
                User's Question: {question}
                Context: {context}
        
                You are new question generator. From following context and user's question, generate additional two questions and answers from context. [question] and [answer] should be in same language as the question follows.
                Generation template must be 'Question: [question] | Answer: [answer] >< Question: [question] | Answer: [answer]'. 
            """
        super().__init__(retriever_fn, llm, template, output_parser)


    def __call__(self, query):
        output = super().__call__(query)
        questions = []
        answers = []
        for i in output.split("><"):
            question, answer = i.split("|")
            questions.append(question.replace("Question:", "").strip())
            answers.append(answer.replace("Answer:", "").strip())
        
        return questions, answers
        