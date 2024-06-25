# 사용법 관련 예제코드 -> 주어진 데이터를 SumRAG에 사용할 수 있는 형태로 변환하는 코드입니다.

from dotenv import load_dotenv
from SumRAG import LLMs
from SumRAG.documents import SumInput
load_dotenv()


with open("./example/labor_standards_act.txt", "r") as f:
    text = f.read()


chapters = {}

new_line = ""
for line in text.split("\n"):
    if line == "":
        continue
    else:
        if line[:len("CHAPTER")] == "CHAPTER":
            if new_line != "":
                chapters[chaper_name].append(new_line)
                new_line = ""

            chaper_name = " ".join(line.split(" ")[2:])
            chapters[chaper_name] = []
        
        else:
            if line[:len("Article")] == "Article":
                if new_line != "":
                    chapters[chaper_name].append(new_line)
                new_line = f"#{line}"

            else:
                new_line += "\\n" + line
            
chapters[chaper_name].append(new_line)

for key in chapters.keys():
    with open(f"./example/txt/{key}.txt", "w") as f:
        f.write("\n".join(chapters[key]))


documents = SumInput.load_from_files("./example/txt", LLMs.gpt3_5)
documents.save("./example/documents")