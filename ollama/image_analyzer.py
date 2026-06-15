import os
import time
import base64
import requests
import shutil
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
from dotenv import load_dotenv
from pathlib import Path

from langchain_ibm import WatsonxEmbeddings
from langchain_ibm import ChatWatsonx
from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

####################
# 모델
####################
load_dotenv()

apiKey = os.getenv("WATSONX_API_KEY")
project_id = os.getenv("WATSONX_PROJECT_ID")
watsonx_ai_url = os.getenv("WATSONX_URL")
hf_token = os.environ["HF_TOKEN"]
COHERE_API_KEY = os.environ["COHERE_API_KEY"]
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

watson_embedding = WatsonxEmbeddings(
    model_id="ibm/granite-embedding-278m-multilingual",
    url=f"{watsonx_ai_url}",
    api_key=f"{apiKey}",
    project_id=f"{project_id}",
)

watson_llm = ChatWatsonx(
    model_id="ibm/granite-4-h-small",
    url=f"{watsonx_ai_url}",
    api_key=f"{apiKey}",
    project_id=f"{project_id}",
    max_tokens=2000,
    params={"temperature": 0},
)

vision_llm = ChatOllama(model="minimax-m3:cloud", temperature=0)

parser = StrOutputParser()

print("모델 초기화 완료")
print(f"LLM: ibm/granite-4-h-small")
print(f"Embedding: ibm/granite-embedding-278m-multilingual")
print(f"Vision LLM: minimax-m3:cloud")


## 이미지 전처리
def process_image(image_path):
    img = Image.open(image_path)
    # 회색
    img = img.convert("L")
    # 대비 향상
    img = ImageEnhance.Contrast(img).enhance(2.0)
    # 선명도 향상
    img = img.filter(ImageFilter.SHARPEN)
    # 크기지정: 처리 속도 최적화
    img.thumbnail((1024, 1024))

    buffer = BytesIO()
    img.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


## vision llm 텍스트 추출
def extract_text_from_image(image_path):
    img_b64 = process_image(image_path)
    message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:image.jpeg;base64,{img_b64}"},
            },
            {
                "type": "text",
                "text": """이 문서의 텍스트를 추출해주세요
                - 표, 항목, 번호 등 구조를 유지하세요
                - 읽을 수 없는 부분은 [불명확]으로 유지하세요.
                - 이미지 설명 없이 텍스트만 출력하세요
                """,
            },
        ]
    )

    return vision_llm.invoke([message]).content


## vectorstore 저장
def build_vectorstore(texts, metadatas):
    split_docs = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    # Document()
    docs = []
    for text, meta in zip(texts, metadatas):
        chunks = split_docs.split_text(text)
        for i, chunk in enumerate(chunks):
            docs.append(Document(page_content=chunk, metadata={**meta, "chunk_id": i}))

    # 기존 DB 삭제
    db_path = "./db/img_vector"
    if Path(db_path).exists():
        shutil.rmtree(db_path)

    vectorstore = Chroma.from_documents(
        docs, watson_embedding, persist_directory=db_path
    )

    return vectorstore


def format_docs(docs):
    return "\n\n".join(
        f"[출처:{d.metadata.get('source', '?')}]\n{d.page_content}" for d in docs
    )


## rag 제안
def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    rag_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                다음 문서 내용을 참고하여 질문에 답하세요.
                \n문서에 없는 내용은 모른다고 답하세요.
                \n\n 문서내용: {context}
                """,
            ),
            ("human", "{question}"),
        ]
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | watson_llm
        | parser
    )
    return chain


def process_documents(image_paths):
    texts = []
    metas = []
    for path in image_paths:
        image_path = getattr(path, "path", None) or getattr(path, "name", None) or path
        # 이미지에서 텍스트 생성
        text = extract_text_from_image(image_path)
        texts.append(text)
        metas.append({"source": image_path})

    # 벡터스토어 저장
    return build_vectorstore(texts=texts, metadatas=metas)


if __name__ == "__main__":
    images = ["./image/receipt1.jpg", "./image/receipt2.jpg", "./image/receipt3.jpg"]
    vectorstore = process_documents(images)
    rag_chain = build_rag_chain(vectorstore)

    questions = ["금액이 총 얼마인가요?", "날짜가 언제인가요?"]
    for q in questions:
        print(f"Q: {q}")
        print(f"A: {rag_chain.invoke(q)}\n")
