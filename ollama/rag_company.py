
# # 사용자가 입력하는 단어와 유사한 2개의 단어 추출
# csv 파일

import os
import shutil
import stat

from dotenv import load_dotenv
import gradio as gr
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, UnstructuredExcelLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


from langchain_ibm import ChatWatsonx, WatsonxEmbeddings
from pathlib import Path

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
  max_tokens = 2000
)

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

LOADERS = {
  ".pdf": PyPDFLoader,
  ".docx": UnstructuredWordDocumentLoader,
  ".xlsx": UnstructuredExcelLoader,
  ".txt": TextLoader
}
CHROMA_DIR = "./db/chroma_db2"
COLLECTION_NAME = "job_rag"
CHUNK_PATH = "./db/chunks.pkl"

DOCUMENTS = []
CHUNKS = []
VECTORSTORE = None

def extract_metadata(file_path): 
  # 2026 상 삼성E&A 직무기술서
  # {YEADR, recruitment_period, company, file_name}
  name = file_path.stem #파일명
  datas = name.split(" ")

  return {
    "year": int(datas[0]),
    "recruitment_period": f"{datas[1]}반기",
    "company":datas[2],
    "document_type": datas[3],
    "file_name": name
  }

def upload_filed(files):
  """
  여러 개의 파일이 업로드 될 때 각 파일을 load()한 결과 DOCUMENTS에 담음
  확장자 분리
  """
  global DOCUMENTS
  all_docs = []

  for file in files:
    path = Path(file.name)
    ext = path.suffix.lower()
    loader = LOADERS[ext](file.name)
    docs = loader.load()

    # metadata 정리
    meta_info = extract_metadata(path)
    # metadata 업데이트
    for doc in docs: 
      doc.metadata.update(meta_info)

    # append, extend
    # append는 리스트를 통째로 하나 추가하고, extend는 리스트 안의 요소들을 하나씩 펼쳐서 추가합니다.

    all_docs.extend(docs)

  DOCUMENTS = all_docs

  return f"문서 수:{len(all_docs)}"

def preview_chunks():
  global DOCUMENTS
  global CHUNKS
  
  if not DOCUMENTS:
    return "문서없음"
  # 전체문서는 documents에 있음
  # 분리 spilitter
  # 청크 10개까지만 출력

  CHUNKS = splitter.split_documents(DOCUMENTS)

  preview = []
  for i, chunk in enumerate(CHUNKS[:10]):
    preview.append(f"""[CHUNK {i+1}]{chunk.page_content[:100]}""")

  return "\n\n".join(preview)


def remove_readonly(func, path, excinfo):
    # 읽기 전용 속성으로 인한 에러일 경우 쓰기 권한을 부여하고 다시 시도
    os.chmod(path, stat.S_IWRITE)
    func(path)

def build_vectorstore():
  global VECTORSTORE
  global CHUNKS

  if not CHUNKS:
    return "먼저 CHUNK를 생성하세요"
  
  # 기존의 벡터 스토어가 있다면 제거
  if Path(CHROMA_DIR).exists():
    shutil.rmtree(CHROMA_DIR, ignore_errors=False, onerror=remove_readonly)

  VECTORSTORE = Chroma.from_documents(
    documents=CHUNKS,
    embedding=watson_embedding,
    persist_directory=CHROMA_DIR,
    collection_name=COLLECTION_NAME
  )

  return f"""
  생성 완료

  Chunk: {len(CHUNKS)}

  Vector: {VECTORSTORE._collection.count}
  """

# UI


with gr.Blocks() as app:
    # tab3개
    gr.Markdown("# 사내 문서 RAG")
    with gr.Tab("문서관리"):
      files = gr.File(file_count="multiple")
      upload_btn = gr.Button("문서업로드")
      upload_status = gr.Textbox()
      upload_btn.click(fn=upload_filed, inputs=files, outputs=upload_status)
      chunk_btn = gr.Button("청구하기")
      chunk_preview = gr.Textbox()
      chunk_btn.click(preview_chunks, outputs=chunk_preview)
      vector_btn = gr.Button("Vector DB 생성")
      vercot_status = gr.Text()
      vector_btn.click(build_vectorstore, outputs=vercot_status)
    with gr.Tab("검색테스트"):
      pass
    with gr.Tab("RGA채팅"):
      pass


if __name__ == "__main__":
    app.launch()
# %%
