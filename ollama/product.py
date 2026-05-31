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
import time

load_dotenv()

apiKey = os.getenv("WATSONX_API_KEY")
project_id = os.getenv("WATSONX_PROJECT_ID")
watsonx_ai_url = os.getenv("WATSONX_URL")

qwen_llm = ChatOllama(model="qwen3.5:4b", temperature=0)
exaone_llm = ChatOllama(model="exaone3.5:2.4b", temperature=0)
watson_llm = ChatWatsonx(
    model_id="ibm/granite-4-h-small",
    url=f"{watsonx_ai_url}",
    api_key=f"{apiKey}",
    project_id=f"{project_id}",
    max_tokens=2000,
)

# - PydanticOutputParser로 상품 리뷰 분석 파이프라인을 완성.
# - 스키마: sentiment, score, pros(list), cons(list), recommend(bool), reply(str)
# - invoke() 사용 후 속도 확인


class ReviewResult(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    score: float = Field(ge=0.0, le=1.0, description="점수")
    props: list[str] = Field(description="긍정적인점 목록")
    cons: list[str] = Field(description="부정적인 목록")
    recommend: bool = Field(description="추천여부")
    reply: str = Field(description="판매자 입장에서의 고객에게 할 답변")


parser = PydanticOutputParser(pydantic_object=ReviewResult)

system_prompt = """\
리뷰를 분석해서 쇼핑몰을 개선하려합니다.
당신은 유명한 애널리틱스입니다.
쇼핑몰 리뷰를 정교하게 분석해주세요
그리고 꼭 한글로 분석해주세요
"""


prompt = PromptTemplate.from_template(
    system_prompt + "\n리뷰: {review}\n 형식:{format_instructions}"
).partial(format_instructions=parser.get_format_instructions())

model = watson_llm
chain = prompt | model | parser


def review_input(texts):
    # === 기준으로 분리
    reviews = [review.strip() for review in texts.split("===") if review.strip()]
    inputs = [{"review": review} for review in reviews]
    start = time.time()
    results = chain.batch(inputs)
    elapsed = time.time() - start

    output = []
    for i, (result, review) in enumerate(zip(results, reviews), 1):
        emoji = {"positive": "😁", "negative": "🤬", "neutral": "🙂"}[result.sentiment]
        output.append(f"[리뷰 {i}]")
        output.append(f"고객 리뷰 {review[:40]}...]")
        output.append(f"리뷰 감정 {emoji} {result.sentiment}(강도: {result.score:.2f})")
        output.append(f"장점 {", ".join(result.props) if result.props else "없음"}")
        output.append(f"단점 {", ".join(result.cons) if result.props else "없음"}")
        output.append(f"추천 여부 {"👍추천" if result.recommend else "👎비추천"}")
        output.append(f"판매자 답변 {result.reply}")
        output.append("-" * 40)
    output.append(f"소요시간: {elapsed:.2f}sec ({len(reviews)} 개 리뷰)")
    return "\n\n".join(output)


demo = gr.Interface(
    fn=review_input,
    inputs=[
        gr.Textbox(
            lines=20,
            label="리뷰 입력",
            placeholder="여러 리뷰 입력 시 구분자로 ===를 사용하세요",
        ),
    ],
    outputs=gr.Textbox(label="결과"),
    title="리뷰",
    description="리뷰 입력 시 감정, 장단점, 추천 여부를 분석합니다.",
)
demo.launch()
