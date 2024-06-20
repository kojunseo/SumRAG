from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from operator import itemgetter

def get_explain_chain(query, llm):
    # Prompt
    template = '''Following is the contents of chapter. You have to summarize the contents of the chapter. 
    The summary should be in under 4 sentences, and should be concise and clear.
    Also, people can breifly understand the contents of the chapter and it should contains all the important points.
    {context} 
    '''

    prompt = ChatPromptTemplate.from_template(template)
    # Chain
    explain_extract_chain = (
        {'context': RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return explain_extract_chain
