from dotenv import load_dotenv
import os
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
import gradio as gr

load_dotenv()

apiKey = os.getenv("WATSONX_API_KEY")
project_id = os.getenv("WATSONX_PROJECT_ID")
watsonx_ai_url = os.getenv("WATSONX_URL")

credentials = Credentials(
    url=f"{watsonx_ai_url}",
    api_key=f"{apiKey}",
)
client = APIClient(credentials)

model = ModelInference(
    model_id="ibm/granite-4-h-small",
    api_client=client,
    project_id=f"{project_id}",
    params={
        "max_tokens": 10000,
        "temperature": 0.15,
    },
)


def interview_text(genre):

    system_prompt = f"""
    당신은 문학 전문 기자입니다.
    주어진 장르의 특징을 분석하고 분석을 바탕으로 인터뷰 질문을 작성하세요
    """

    user_prompt = f"""
    [장르]
    {genre}
    요구사항
    1. 장르 특징 5줄 정리
    2. 인터뷰 질문 8가지
    """

    messages = [
        {"role": "system", "content": system_prompt},
        # {"role": "user", "content":user_prompt }
    ]

    generated_response = model.chat(messages=messages)
    return generated_response["choices"][0]["message"]["content"]


demo = gr.Interface(
    fn=interview_text,
    inputs=[
        gr.Textbox(label="genre"),
    ],
    outputs=[gr.Markdown()],
    title="🙌 작가 인터뷰 질문 생성 프로그램",
    description="장르 작성 시 AI가 장르의 특징 및 인터뷰 질문을 생성해 드립니다.",
)

demo.launch()
