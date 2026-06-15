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

from langgraph.graph import StateGraph, START, END
from typing import TypedDict

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
watson_embedding = WatsonxEmbeddings(
    model_id="ibm/granite-embedding-278m-multilingual",
    url=f"{watsonx_ai_url}",
    api_key=f"{apiKey}",
    project_id=f"{project_id}",
)


# state 정의
class RagState(TypedDict):
    query: str
    retrieved_docs: list[Document]
    answer: str


# node
def retrieve(state):
    # 기존 백터스토어에 질의
    vectorstore = Chroma(
        collection_name="docs",
        embedding_function=watson_embedding,
        persist_directory="./db/chroma_db",
    )

    docs = vectorstore.similarity_search(state["query"], k=3)
    return {"retrieved_docs": docs}


def generate(state):
    context = "\n\n".join(doc.page_content for doc in state["retrieved_docs"])

    prompt = """\
    다음 컨텍스트를 참고하여 잘문에 대답하세요
    컨텍스트에 없는 내용은 모른다고 답하세요
    
    컨텍스트:
    {context}
    
    질문:
    {query}
    """

    response = watson_llm.invoke(prompt.format(context=context, query=state["query"]))
    return {"answer": response.content}


# 분할 및 백터스토어 저장
def process_pdf(file_input):
    """
    pdf 로드 -> 분할 -> 벡터스토어 저장
    반환: 청크 개수 리턴
    """

    if file_input is None:
        return "PDF 파일을 업로드 해주세요"

    # 문서 로드
    loader = PyPDFLoader(file_input)
    pages = loader.load()

    # 문서 분할
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(pages)

    # 총 청크 수
    chunk_count = len(chunks)

    # 기존 db 존재 한다면 컬렉션 제거
    vectorstore = Chroma(
        collection_name="docs",
        embedding_function=watson_embedding,
        persist_directory="./db/chroma_db",
    )
    try:
        vectorstore.delete_collection()
    except Exception:
        pass

    # 저장 - 새로운 백터스토어 생성
    Chroma.from_documents(
        chunks,
        embedding=watson_embedding,
        collection_name="docs",
        persist_directory="./db/chroma_db",
    )

    return f"총 페이지 수:{chunk_count}"


# 결과 받아오기
def rag_chat(question_input):
    result = app.invoke({"query": question_input})
    return result["answer"]


# 그래프 구성
graph = StateGraph(RagState)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

app = graph.compile()

with gr.Blocks() as demo:
    gr.Markdown("")

    with gr.Tabs():
        with gr.Tab("1단계 - PDF & Chunk 확인"):

            file_input = gr.File(label="PDF 파일", file_types=[".pdf"])
            btn1 = gr.Button("분석 시작")

            output = gr.Textbox(label="처리 결과")

            btn1.click(
                fn=process_pdf,
                inputs=[file_input],
                outputs=[output],
            )

            question_input = gr.Textbox(label="질문 입력")
            run_btn = gr.Button("질문 하기")
            answer_output = gr.Textbox(label="최종답변", lines="10")

            run_btn.click(
                fn=rag_chat,
                inputs=[question_input],
                outputs=[answer_output],
            )


demo.launch()
