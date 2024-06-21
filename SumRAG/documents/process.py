from glob import glob
from langchain_core.documents import Document
from .explain_extract import get_explain_chain
from .keyword_extract import get_keyword_chain
from .inputs import SumInput
import warnings

def load_from_files(path, llm, ext=".txt"):

    warnings.warn("This function is deprecated. Use SumInput.load_from_files instead.")
    explain_extract_chain = get_explain_chain("summarize", llm)
    keyword_extract_chain = get_keyword_chain("keyword", llm)

    files = glob(path + "/*" + ext)
    documents = {}
    keyword_explains = {}
    keyword_document_page = {}

    for file in files:
        with open(file, "r") as f:
            text = f.read()
            text = text.split("\n")
            document = [Document(page_content=t, metadata={"page_index": file, "keyword":t.split("\\n")[0]}) for i, t in enumerate(text)]
        this_explain = explain_extract_chain.invoke({"context": document})
        keyword_extract_chain_output = keyword_extract_chain.invoke({"context": this_explain})
        keyword_explains[keyword_extract_chain_output] = Document(page_content=this_explain, metadata={"page_index": file, "keyword": keyword_extract_chain_output})
        keyword_document_page[keyword_extract_chain_output] = file
        documents[keyword_extract_chain_output] = document

    keywords = list(documents.keys())

    return SumInput(docs=documents, keyword_explains=keyword_explains, keywords=keywords, keyword_page_index=keyword_document_page)