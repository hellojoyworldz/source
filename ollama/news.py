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

qwen_llm = ChatOllama(model="qwen3.5:4b", temperature=0)
exaone_llm = ChatOllama(model="exaone3.5:2.4b", temperature=0)

# - JsonOutputParser를 사용하여 뉴스 기사에서 정보를 추출하는 체인 작성
# - 추출 항목: title(str), date(str), keywords(list), category(str)
# - 3개 이상의 뉴스 텍스트를 batch()로 한번에 처리


class NewsResult(BaseModel):
    title: str
    date: str = Field(description="YYYY-MM-DD 형식, 없는 경우 '없음'")
    keywords: list[str] = Field(description="핵심 키워드 3개 이내")
    category: Literal["정치", "경제", "사회", "문화", "스포츠", "IT", "국제", "기타"]


pydantic_parser = PydanticOutputParser(pydantic_object=NewsResult)

system_prompt = """\
당신은 뉴스 분석가입니다. 반드시 JSON 형태로 출력해주세요.
뉴스 기사의 정보를 추출하여 요약을 해줍니다
기사에서 아래 정보를 추출하세요
아래의 형식처럼 요약해주세요

주의사항
- date 없으면 없음 표시
- date YYYY-MM-DD 형식 무조건 
"""


prompt = PromptTemplate.from_template(
    system_prompt + "\n기사: {question}\n{format_instructions}"
).partial(format_instructions=pydantic_parser.get_format_instructions())

model = exaone_llm
parser = JsonOutputParser()
chain = prompt | model | parser


def news_input(texts):
    # === 기준으로 분리
    articles = [article.strip() for article in texts.split("===") if article.strip()]
    inputs = [{"question": article} for article in articles]
    response = chain.batch(inputs)
    return "\n\n".join(str(item) for item in response)


demo = gr.Interface(
    fn=news_input,
    inputs=[
        gr.Textbox(
            lines=20,
            label="뉴스 기사1",
            placeholder="여러 기사 입력 시 구분자로 ===를 사용하세요",
        ),
    ],
    outputs=gr.Textbox(label="결과"),
    title="뉴스",
    description="뉴스 기사 요약",
)
demo.launch()
