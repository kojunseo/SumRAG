import os
import ojson
from glob import glob
from langchain_core.documents import Document


class SumInput:
    def __init__(self, docs, keyword_explains, keywords, keyword_page_index):
        self.docs = docs
        self.__keyword_explains = keyword_explains
        self.keywords = keywords
        self.keyword_page_index = keyword_page_index


    @property
    def keyword_explains(self):
        return list(self.__keyword_explains.values())
    

    def save(self, path):
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
                            {"meta": doc.page_content.split("\n")[0],
                            "content": "\n".join(doc.page_content.split("\n")[1:])} for doc in self.docs[keyword]
                        ]
                    }
                ,f)

    @classmethod
    def load(cls, path):
        docs = {}
        keyword_explains = {}
        keywords = []
        keyword_page_index = {}
        for file in glob(os.path.join(path, "*.json")):
            with open(file) as f:
                data = ojson.load(f)
                keyword = data["meta"]["keyword"]
                keywords.append(keyword)
                keyword_explains[keyword] = Document(page_content=data["meta"]["explain"], metadata={"keyword": keyword})
                keyword_page_index[keyword] = data["meta"]["page_label"]
                docs[keyword] = [Document(page_content=content["meta"] + "\n" + content["content"]) for content in data["content"]]
        
        return cls(docs, keyword_explains, keywords, keyword_page_index)
