from langchain_ollama import ChatOllama
from langchain_ibm import ChatWatsonx
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
)
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
import os
import gradio as gr

load_dotenv()

apiKey = os.getenv("WATSONX_API_KEY")
project_id = os.getenv("WATSONX_PROJECT_ID")
watsonx_ai_url = os.getenv("WATSONX_URL")

qwen_llm = ChatOllama(model="qwen3.5:4b")
exaone_llm = ChatOllama(model="exaone3.5:2.4b")
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
    [("system", system_prompt), ("human", "{question}")]
)
model = exaone_llm
parser = StrOutputParser()
chain = prompt | model | parser


def chat(question, history):
    chat_history = []

    for msg in history:
        if msg["role"] == "user":
            chat_history.append(("human", msg["content"]))
        elif msg["role"] == "assistant":
            chat_history.append(("ai", msg["content"]))

    response = chain.invoke({"history": history, "question": question})
    return response


demo = gr.ChatInterface(fn=chat, title="요리전문가", description="요리경력 20년 전문가")

demo.launch()
