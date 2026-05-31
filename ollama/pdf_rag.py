# 라이브러리 로드
from langchain_ollama import ChatOllama
from langchain_ibm import ChatWatsonx
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
)
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
)
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.chat_history import (
    InMemoryChatMessageHistory,
    BaseChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    WebBaseLoader,
    DirectoryLoader,
)
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ibm import WatsonxEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
import gradio as gr

# .env 내용 가져오기
load_dotenv()

apiKey = os.getenv("WATSONX_API_KEY")
project_id = os.getenv("WATSONX_PROJECT_ID")
watsonx_ai_url = os.getenv("WATSONX_URL")
hf_token = os.environ["HF_TOKEN"]

# 모델 (LLM, Embedding)
qwen_llm = ChatOllama(model="qwen3.5:4b")
exaone_llm = ChatOllama(model="exaone3.5:2.4b")
watson_llm = ChatWatsonx(
    model_id="ibm/granite-4-h-small",
    url=f"{watsonx_ai_url}",
    api_key=f"{apiKey}",
    project_id=f"{project_id}",
    max_tokens=2000,
)

# 핵심 개념

# - Retriever: Vector Store를 LCEL 체인에 연결하는 인터페이스. as_retriever()로 생성.
# - RunnablePassthrough: 질문 원문을 context와 함께 LLM에 전달하기 위해 사용.
# - context + question → prompt → LLM → 답변 흐름이 기본 RAG 패턴.
# - search_kwargs={'k': 4}: 검색할 유사 문서 수 — 많을수록 정확하지만 토큰 증가.

# 💡 과제 1
# - PyPDFLoader로 PDF 파일을 로드하고 총 페이지 수와 첫 페이지 내용을 출력.
# - RecursiveCharacterTextSplitter로 chunk_size 300 / overlap 30 으로 분할.
# - 분할된 청크 수와 첫 번째 청크의 내용 및 metadata를 출력.

ollama_embedding = OllamaEmbeddings(model="nomic-embed-text-v2-moe")


def analytics(file_input):
    if file_input is None:
        return ("PDF 파일을 업로드 해주세요", "", "", "", "")

    # 문서 로드
    loader = PyPDFLoader(file_input)
    pages = loader.load()

    # 총 페이지 수, 첫 페이지 내용
    total_pages = len(pages)
    first_pages = pages[0].page_content[:1000]

    # 문서 분할
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(pages)

    # 총 청크 수, 첫 번째 청크 메타데이터
    chunk_count = len(chunks)
    first_chunk = chunks[0].page_content
    first_chunk_meta = chunks[0].metadata

    return (total_pages, first_pages, chunk_count, first_chunk, first_chunk_meta)


def qna_analytics(file_input, q_input):
    if file_input is None:
        return ("PDF 파일을 업로드 해주세요", "")

    loader = PyPDFLoader(file_input)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    split_docs = splitter.split_documents(docs)

    # 임베딩
    faiss_store = FAISS.from_documents(documents=split_docs, embedding=ollama_embedding)

    # 검색
    retriever = faiss_store.as_retriever(search_kwargs={"k", 3})
    retriever_docs = retriever.invoke(q_input)

    # context생성
    context = "\n\n".join([doc.page_context for doc in retriever_docs])

    # LLM
    system_message = """\당신은 pdf 기반 RAG 입니다.

    문서:
    {context}

    질문:
    {question}
    """
    rag_prompt = ChatPromptTemplate.from_template(message=system_message)
    chain = rag_prompt | watson_llm | StrOutputParser()


with gr.Blocks() as demo:
    gr.Markdown("")

    with gr.Tabs():
        with gr.Tab("1단계 - PDF & Chunk 확인"):

            file_input = gr.File(label="PDF 파일", file_types=[".pdf"])
            btn1 = gr.Button("분석 시작")
            result_box = gr.Textbox(label="분석 결과")
            total_pages_box = gr.Textbox(label="총 페이지 수")
            first_page_box = gr.Textbox(label="첫 페이지 내용", lines=10)
            chunk_count_box = gr.Textbox(label="청크 수")
            first_chunk_meta_box = gr.Textbox(label="첫 청크 매타")
            # first_chunk_box = gr.Textbox(label="첫 청크 내용", lines=10)

            btn1.click(
                fn=analytics,
                inputs=[file_input],
                outputs=[
                    result_box,
                    total_pages_box,
                    first_page_box,
                    chunk_count_box,
                    first_chunk_meta_box,
                    # first_chunk_box,
                ],
            )

        with gr.Tab("2단계 - RAG QA"):
            file_input2 = gr.File(label="PDF 파일", file_types=[".pdf"])
            q_input2 = gr.Textbox(label="질문 입력")
            btn2 = gr.Button("질문 하기")
            searched_chunk_box = gr.Textbox(label="검색된 청크", lines="10")
            result_box = gr.Text(label="최종 결과", lines="10")

            btn2.click(
                fn=qna_analytics,
                inputs=[file_input2, q_input2],
                outputs=[searched_chunk_box, result_box],
            )


demo.launch()
