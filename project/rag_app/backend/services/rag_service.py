import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import (
    ChatPromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser

from backend.ai.llm import watson_llm
from backend.ai.embedding import watson_embedding

UPLOAD_PATH = "uploads"
DB_PATH = "./db/vectorstore"


def upload_document(file):
    # file 저장 - 서버 컴퓨터에 저장
    file_path = os.path.join(UPLOAD_PATH, file.filename)

    with open(file_path, "wb") as f:
        f.write(file.file.read())

    # pdf 업로드 -> 분할 -> 인덱스 생성 -> 백터스토어 저장
    # pdf 로드
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # 문서 분할
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(pages)

    # 저장
    faiss_store = FAISS.from_documents(documents=chunks, embedding=watson_embedding)
    faiss_store.save_local(DB_PATH)

    return {"message": "업로드 성공"}


# 질문 -> 유사도 검색 -> 찾은 문서 기반으로 -> LLM에게 답변 받음
def rag_chat(question: str):
    # 벡터스토어에서 불러오기
    faiss_store = FAISS.load_local(
        DB_PATH, watson_embedding, allow_dangerous_deserialization=True
    )

    # retriever (Document타입)
    retriever = faiss_store.as_retriever(search_kwargs={"k": 3})

    # context(retriever)를 LLM에게 보내서, 질문에 대한 답변을 받아야 한다
    system_message = """당신은 pdf 기반 RAG 입니다.

    문서:
    {context}

    질문:
    {question}
    """

    rag_prompt = ChatPromptTemplate.from_template(system_message)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | watson_llm
        | StrOutputParser()
    )
    answer = chain.invoke(question)
    return answer


# SSE
async def rag_chat_stream(question: str):

    faiss_store = FAISS.load_local(
        DB_PATH, watson_embedding, allow_dangerous_deserialization=True
    )

    retriever = faiss_store.as_retriever(search_kwargs={"k": 3})

    system_message = """당신은 pdf 기반 RAG 입니다.

    문서:
    {context}

    질문:
    {question}
    """

    rag_prompt = ChatPromptTemplate.from_template(system_message)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | watson_llm
        | StrOutputParser()
    )

    # chain.stream(): 동기방식(요청 -> 응답 할 때 까지 기다리는 방식)
    # chain.astream(): 비동기방식 (다른 일을 할 수 있음)
    async for chunk in chain.astream(question):
        yield chunk
