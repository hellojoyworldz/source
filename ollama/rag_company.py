# # 사용자가 입력하는 단어와 유사한 2개의 단어 추출
# csv 파일

import os
import shutil
import stat
import pickle

from dotenv import load_dotenv
import gradio as gr
from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_ibm import ChatWatsonx, WatsonxEmbeddings
from pathlib import Path

from langchain_classic.chains.query_constructor.base import AttributeInfo
from langchain_classic.retrievers import (
    EnsembleRetriever,
    ContextualCompressionRetriever,
    BM25Retriever,
)
from langchain_classic.retrievers.self_query.chroma import ChromaTranslator
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_cohere import CohereRerank

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_classic.chains import ConversationalRetrievalChain

load_dotenv()

apiKey = os.getenv("WATSONX_API_KEY")
project_id = os.getenv("WATSONX_PROJECT_ID")
watsonx_ai_url = os.getenv("WATSONX_URL")
hf_token = os.environ["HF_TOKEN"]
COHERE_API_KEY = os.environ["COHERE_API_KEY"]

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
)

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

LOADERS = {
    ".pdf": PyPDFLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".txt": TextLoader,
}
CHROMA_DIR = "./db/chroma_db2"
COLLECTION_NAME = "job_rag"
CHUNK_PATH = "./db/chunks.pkl"

DOCUMENTS = []
CHUNKS = []
VECTORSTORE = None

BM25_RETRIEVER = None
DENSE_RETRIEVER = None
SELFQUERY_RETRIEVER = None
FINAL_RETRIEVER = None

QA_CHAIN = None

META_FIELDS = [
    AttributeInfo(name="year", description="채용연도", type="int"),
    AttributeInfo(
        name="recruitment_period", description="상반기 또는 하반기", type="string"
    ),
    AttributeInfo(name="company", description="회사명", type="string"),
    AttributeInfo(
        name="document_type",
        description="직무기술서, 채용공고, 기업분성",
        type="string",
    ),
    AttributeInfo(
        name="file_name",
        description="파일명",
        type="string",
    ),
]

# 대화 메모리
# ConversationBufferWindowMemory: 최근 N개의 대화만 기억하는 창(Window)
memory = ConversationBufferWindowMemory(
    k=5, memory_key="chat_history", return_messages=True, output_key="answer"
)

SYSTEM_PROMPT = """당신은 회사 내부 문서를 기반으로 직원들의 질문에 대답한는 AI 어시스턴트입니다.

다음 규칙을 반드시 지켜주세요.
1. 제공된 문서 내용에만 기반하여 답변하세요.
2. 문서에 없는 내용은 '해당 내용은 제공된 문서에서 찾을 수 없습니다' 라고 답변하세요.
3. 답변 마지막에 참고한 문서명을 명시하세요.
4. 한국어로 명확하고 구체적으로 답변하세요.
"""
QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        # MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            """
            [참고문서] 
            {context} 
            
            [질문] 
            {question}
            
            
            질문이 들어오면, 참고문서에서 내용을 찾아 설명해주세요.
            """,
        ),
    ]
)


# ==========
# 앱 시작 시
# ==========
def build_retriever(chunks, save_chunks=False):

    global BM25_RETRIEVER
    global DENSE_RETRIEVER
    global SELFQUERY_RETRIEVER
    global FINAL_RETRIEVER
    global VECTORSTORE

    # 검색테스트 탭으로 바로 시작 한다면

    # BM25 index 작업을 폴더에 저장
    if save_chunks:
        with open(CHUNK_PATH, "wb") as f:
            pickle.dump(chunks, f)

    # retriever 초기화
    # BM25 index는 Chroma에 저장되지 않음
    BM25_RETRIEVER = BM25Retriever.from_documents(chunks, k=5)

    # 일반검색
    DENSE_RETRIEVER = VECTORSTORE.as_retriever(k=20)

    # 셀프쿼리
    SELFQUERY_RETRIEVER = SelfQueryRetriever.from_llm(
        llm=watson_llm,
        vectorstore=VECTORSTORE,
        document_contents="계열사 직무 기술서 문서",
        metadata_field_info=META_FIELDS,
        structured_query_translator=ChromaTranslator(),
        search_kwargs={"k": 20},
    )

    # final: bm25 + 일반 dense
    ensemble = EnsembleRetriever(
        retrievers=[BM25_RETRIEVER, DENSE_RETRIEVER], weights=[0.35, 0.65]
    )
    reranker = CohereRerank(model="rerank-v4.0-pro", top_n=5)
    FINAL_RETRIEVER = ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=ensemble
    )

    return "Retriever 생성 완료"


def initialize():
    global VECTORSTORE

    # db 없는 경우
    if not Path(CHUNK_PATH).exists():
        print("기존 vector 없음")
        return

    # BM25 제외, retriever는 이 부분만 하면 가능
    # 기존 vectorstore호출
    VECTORSTORE = Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=watson_embedding,
    )

    # bm25 파일 처리
    if Path(CHUNK_PATH).exists():

        with open(CHUNK_PATH, "rb") as f:
            chunks = pickle.load(f)

            build_retriever(chunks=chunks, save_chunks=False)
            print("Retriever 로드")


# ==========
# Tab1 기능 구현
# ==========
def extract_metadata(file_path):
    # 2026 상 삼성E&A 직무기술서
    # {YEAR, recruitment_period, company, file_name}
    name = file_path.stem  # 파일명
    datas = name.split(" ")

    return {
        "year": int(datas[0]),
        "recruitment_period": f"{datas[1]}반기",
        "company": datas[2],
        "document_type": datas[3],
        "file_name": name,
    }


def upload_filed(files):
    """
    여러 개의 파일이 업로드 될 때 각 파일을 load()한 결과 DOCUMENTS에 담음
    확장자 분리
    """

    global DOCUMENTS
    global CHUNKS
    global VECTORSTORE
    global BM25_RETRIEVER
    global DENSE_RETRIEVER
    global SELFQUERY_RETRIEVER
    global FINAL_RETRIEVER

    BM25_RETRIEVER = None
    DENSE_RETRIEVER = None
    SELFQUERY_RETRIEVER = None
    FINAL_RETRIEVER = None
    CHUNKS = []
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


def build_vectorstore():
    global VECTORSTORE
    global CHUNKS

    if not CHUNKS:
        return "먼저 CHUNK를 생성하세요"

    # 기존 연결이 있으면 먼저 닫은 뒤 같은 경로를 다시 사용
    if VECTORSTORE is not None:
        try:
            VECTORSTORE._client.close()
        except Exception:
            pass
        VECTORSTORE = None

    # 기존의 벡터 스토어가 있다면 제거
    if Path(CHROMA_DIR).exists():
        shutil.rmtree(CHROMA_DIR)

    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)

    VECTORSTORE = Chroma.from_documents(
        documents=CHUNKS,
        embedding=watson_embedding,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )

    # retriever 생성\
    # save_chunks=True: bm25 index 저장
    build_retriever(CHUNKS, save_chunks=True)

    global QA_PROMPT
    QA_CHAIN = None

    return f"""
  생성 완료

  Chunk: {len(CHUNKS)}

  Vector: {VECTORSTORE._collection.count}
  """


# ==========
# Tab2 기능 구현
# 1. 임베딩 작업 완료
# 2. 문서관리 -> 검색테스트
# ==========
def format_docs(docs):
    """Document 객체에서 page_content 추출"""

    if not docs:
        return "검색 결과 없음"

    result = []
    result.append(f"검색 결과 수 {len(docs)}건\n")

    for i, d in enumerate(docs, 1):
        result.append(f"""\
[문서 {i}]

회사: {d.metadata.get("company", "")}
유형: {d.metadata.get("document_type", "-")}
년도: {d.metadata.get("year", "")} {d.metadata.get("recruitment_period", "-")}
출처: {d.metadata.get("company", "")}
내용: {d.page_content}
""")

    return "\n\n".join(result)


def search_test(query):
    global CHUNKS
    if FINAL_RETRIEVER is None:
        return (
            "BM25 retriever 미생성",
            "Dense retriever 미생성",
            "Selfquery retriever 미생성",
            "Final retriever 미생성",
        )

    # 각각의 retriever 결과 추출 한 후
    # format_docs return

    bm25_docs = format_docs(BM25_RETRIEVER.invoke(query))
    dense_docs = format_docs(DENSE_RETRIEVER.invoke(query))
    self_docs = format_docs(SELFQUERY_RETRIEVER.invoke(query))
    final_docs = format_docs(FINAL_RETRIEVER.invoke(query))

    return bm25_docs, dense_docs, self_docs, final_docs


# ==========
# Tab3 기능 구현
# gradio chat interface
# - history: 대화 이력 관리
# RunnableWithMessageHistory
# ==========


def create_chain():
    global QA_CHAIN

    if QA_CHAIN is None:
        QA_CHAIN = ConversationalRetrievalChain.from_llm(
            llm=watson_llm,
            retriever=FINAL_RETRIEVER,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            return_source_documents=True,
        )

    return QA_CHAIN


def chat(message, history):
    global QA_CHAIN

    if FINAL_RETRIEVER is None:
        return "먼저 vector db를 생성하세요"

    QA_CHAIN = create_chain()
    response = QA_CHAIN.invoke({"question": message})

    answer = response["answer"]

    sources = []
    for doc in response["source_documents"]:
        sources.append(
            f"{doc.metadata.get("company", "-")} - "
            f"{doc.metadata.get("file_name","-")}"
        )
    answer += "\n\n[참고문서]\n"
    answer += "\n".join(list(set(sources)))

    return answer


# ==========
# UI
# ==========
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
        query = gr.Text(label="검색어")
        search_btn = gr.Button("검색")
        bm25_box = gr.Textbox(label="bm25")
        dense_box = gr.Textbox(label="dense")
        self_box = gr.Textbox(label="self")
        rerank_box = gr.Textbox(label="final")
        search_btn.click(
            fn=search_test,
            inputs=query,
            outputs=[bm25_box, dense_box, self_box, rerank_box],
        )
    with gr.Tab("RGA채팅"):
        reg_btn = gr.Button("RAG생성")
        textbox = gr.Textbox()
        chatbot = gr.ChatInterface(fn=chat)
        reg_btn.click()

if __name__ == "__main__":
    initialize()
    app.launch()
