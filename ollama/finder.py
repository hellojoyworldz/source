# 사용자가 입력하는 단어와 유사한 2개의 단어 추출
# csv 파일

import os

from dotenv import load_dotenv
import gradio as gr
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS


from langchain_ibm import ChatWatsonx, WatsonxEmbeddings

load_dotenv()

apiKey = os.getenv("WATSONX_API_KEY")
project_id = os.getenv("WATSONX_PROJECT_ID")
watsonx_ai_url = os.getenv("WATSONX_URL")
hf_token = os.environ["HF_TOKEN"]
COHERE_API_KEY = os.environ["COHERE_API_KEY"]

watson_embedding = WatsonxEmbeddings(
    model_id="ibm/granite-embedding-278m-multilingual",
    url = f"{watsonx_ai_url}",
    api_key = f"{apiKey}",
    project_id=f"{project_id}"
)

watson_llm = ChatWatsonx(
  model_id="ibm/granite-4-h-small",
  url=f"{watsonx_ai_url}",
  api_key = f"{apiKey}",
  project_id=f"{project_id}",
  max_tokens = 2000,
  params = {
    "temperature":0
  }
)


# csv loader
csv_loader = CSVLoader(
  "./data/myData.csv",  
  encoding="utf-8", 
  csv_args={
    'delimiter':',', 
    "fieldnames": ["Words"]
  }
)
csv_docs = csv_loader.load()

# 백터 메모리
vectorstore = FAISS.from_documents(
    documents=csv_docs,
    embedding=watson_embedding,
    collection_name="finder_words",
)


def find_simmilar(query):
  if not query.strip():
     return "검색어 확인",""
  
  docs = vectorstore.similarity_search(query, k=2)
  result1 = docs[0].page_content if len(docs) > 1 else ""
  result2 = docs[1].page_content if len(docs) > 2 else ""
  return result1, result2
  


with gr.Blocks() as demo:
    gr.Markdown("# Educate Kids")
    gr.Markdown("## 비슷한 단어 또는 문장을 찾아드립니다.")
    query = gr.Textbox(label="단어 입력", placeholder="Apple")
    btn = gr.Button("Find Similar Things")
    output1 = gr.Textbox(label="TopMatch1")
    output2 = gr.Textbox(label="TopMatch2")
    btn.click(find_simmilar, inputs=query, outputs=[output1, output2])

demo.launch()