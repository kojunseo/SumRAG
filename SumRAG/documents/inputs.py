import os
import ojson
from glob import glob
from .explain_extract import get_explain_chain
from .keyword_extract import get_keyword_chain
from langchain_core.documents import Document


class SumInput:
    """
    This class is used to load and save the documents and keyword explains. 
    Detail of the class is in the `SumRAG` documentation : https://github.com/kojunseo/SumRAG
    Example:
        .. code-block:: python
        from SumRAG.documents import SumInput
        # Load the txt files and extract the keyword explains.
        documents = SumInput.load_from_files("./data_txt", llm3) 

        # Automatically creates the directory and save the changed documents.
        documents.save("./data_to_json") 

        # Load the saved documents.
        documents = SumInput.load("./data_to_json")
    
    Returns:
        `SumInput`: The SumInput object.
    """
    def __init__(self, docs, keyword_explains, keywords, keyword_page_index):
        """
        Initialize the SumInput class. Recommended to use the `load_from_files` to make class.
        Parameters:
            docs (Dict[str, List[Document]]): A dictionary of documents with the keyword as the key.
            keyword_explains (Dict[str, Document]): A dictionary of keyword explains with the keyword as the key.
            keywords (List[str]): A list of keywords.
            keyword_page_index (Dict[str, str]): A dictionary of keyword and the page index.
        """
        self.docs = docs
        self.__keyword_explains = keyword_explains
        self.keywords = keywords
        self.keyword_page_index = keyword_page_index


    @property
    def keyword_explains(self):
        return list(self.__keyword_explains.values())
    

    def save(self, path):
        """
        Save the documents and keyword explains under the path as json format.
        Parameters:
            `path (str)`: The path to save the documents.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        for keyword in self.keywords:
            with open(os.path.join(path, keyword + ".json"), "w") as f:
                ojson.dump(
                    {
                        "meta": {
                            "keyword":keyword,
                            "explain": self.__keyword_explains[keyword].page_content,
                            "page_label": self.keyword_page_index[keyword]
                        },
                        "content":[
                            {"meta": doc.page_content.split("\\n")[0],
                            "content": "\\n".join(doc.page_content.split("\\n")[1:])} for doc in self.docs[keyword]
                        ]
                    }
                ,f)

    @classmethod
    def load(cls, path):
        """
        Load the documents and keyword explains from the path. To create the json files, use the `save` method.
        """
        docs = {}
        keyword_explains = {}
        keywords = []
        keyword_page_index = {}
        files = glob(os.path.join(path, "*.json"))

        assert len(files) > 0, "No files found in the directory"
        for file in files:
            with open(file) as f:
                data = ojson.load(f)
                keyword = data["meta"]["keyword"]
                keywords.append(keyword)
                keyword_explains[keyword] = Document(page_content=data["meta"]["explain"], metadata={"keyword": keyword})
                keyword_page_index[keyword] = data["meta"]["page_label"]
                docs[keyword] = [Document(page_content=content["meta"] + "\n" + content["content"], metadata={"keyword":content["meta"]}) for content in data["content"]]
        
        return cls(docs, keyword_explains, keywords, keyword_page_index)


    @classmethod
    def load_from_files(cls, path, llm, ext=".txt"):
        """
        Load the documents from the path and extract the keyword explains using the llm.
        Parameters:
            `path (str)`: The path to load the documents.
            `llm (SumRAG.LLMs -> llms properties)`: The llm to extract the keyword explains.
            `ext (str)`: The extension of the files to load.
        """
        explain_extract_chain = get_explain_chain(llm)
        keyword_extract_chain = get_keyword_chain(llm)

        files = glob(path + "/*" + ext)
        assert len(files) > 0, "No files found in the directory"
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

        return cls(docs=documents, keyword_explains=keyword_explains, keywords=keywords, keyword_page_index=keyword_document_page)