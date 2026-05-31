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

qwen_llm = ChatOllama(model="qwen3.5:4b")
exaone_llm = ChatOllama(model="exaone3.5:2.4b")


# 과제 1

# - ChatOllama로 '요리 전문가' 페르소나 챗봇을 만드세요
# - ChatPromptTemplate으로 system에 역할, human에 {question} 변수를 정의
# - StrOutputParser로 텍스트만 추출하여 깔끔하게 출력


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
    response = chain.invoke({"question": question})
    return response


demo = gr.ChatInterface(fn=chat, title="요리전문가", description="요리경력 20년 전문가")

demo.launch()
