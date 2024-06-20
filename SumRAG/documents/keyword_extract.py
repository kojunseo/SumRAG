from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from operator import itemgetter

def get_keyword_chain(query, llm):
    # Prompt
    template = '''Following is the summary of the chapter. You have give a main keyword of the chapter.
    You have to return a single keyword under 2 or 3 words.
    {context}
    '''

    prompt = ChatPromptTemplate.from_template(template)
    # Chain
    keyword_extract_chain = (
        {'context': RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return keyword_extract_chain