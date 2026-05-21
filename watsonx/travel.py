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
    params={"max_tokens": 1000},
)


def recommend(region, budget, theme, duration):

    system_prompt = f"""
    당신은 여행 전문가입니다. 
    반드시
    1. 일정표
    2. 추천장소
    3. 맛집
    4. 예상비용

    을 포함해줘
    """

    user_prompt = f"""
    여행 지역은 {region}이고, 예산은 {budget}만원입니다. 여행 테마는 {theme}이며, 여행 기간은 {duration}입니다. 이 정보를 바탕으로 여행 일정을 추천해주세요.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # generated_response = model.chat(messages=messages)
    # return generated_response["choices"][0]["message"]["content"]

    # chat_stream()
    generated_response = model.chat_stream(messages=messages)
    full_response = ""
    for chunk in generated_response:
        if chunk["choices"]:
            full_response += chunk["choices"][0]["delta"].get("content", "")
            yield full_response


demo = gr.Interface(
    fn=recommend,
    inputs=[
        gr.Text(label="여행 지역"),
        gr.Slider(100, 300, label="예산 '(기간))"),
        gr.Dropdown(["모험", "휴향", "음식"]),
        gr.Radio(["1일", "2~3일", "4~7일", "1주 이상"], label="여행 기간"),
    ],
    outputs=[gr.Markdown()],
    title="여행 추천 프로그램",
    description="텍스트 입력 시 ai 요약",
)

demo.launch()
