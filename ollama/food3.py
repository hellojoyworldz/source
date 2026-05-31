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
from langchain_core.chat_history import (
    InMemoryChatMessageHistory,
    BaseChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
import os
import gradio as gr
import uuid

load_dotenv()

apiKey = os.getenv("WATSONX_API_KEY")
project_id = os.getenv("WATSONX_PROJECT_ID")
watsonx_ai_url = os.getenv("WATSONX_URL")

store = {}


def get_session_history(session_id) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def create_chain():

    watson_llm = ChatWatsonx(
        model_id="ibm/granite-4-h-small",
        url=f"{watsonx_ai_url}",
        api_key=f"{apiKey}",
        project_id=f"{project_id}",
        max_tokens=2000,
    )

    system_prompt = """\
    당신은 20년 경력의 전문 쉐프이자 요리 연구가
    사용자의 요리 질문에 대해
    재료, 조리법, 실패 방지 팁 등을 포함하여
    항상 한국어로 답변하세요
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )
    model = watson_llm
    parser = StrOutputParser()
    chain = prompt | model | parser
    with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    return with_history


chain = create_chain()


def chat(question, history, session_id):
    full_response = ""
    for chunk in chain.stream(
        {"question": question}, config={"configurable": {"session_id": session_id}}
    ):
        full_response += chunk
        yield full_response


with gr.Blocks() as chatbot:
    # 세션ID uuid 사용
    # gr.State(): 사용자별 데이터를 서버 메모리에 저장하는 커포넌트
    session_state = gr.State(str(uuid.uuid4()))
    gr.ChatInterface(
        fn=chat,
        additional_inputs=[session_state],
        title="요리전문가",
        description="요리경력 20년 전문가",
    )

chatbot.launch()
