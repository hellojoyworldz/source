# 멀티 모달: 이미지 + 텍스트
# 일반 LLM: 텍스트

from dotenv import load_dotenv
import os
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
import gradio as gr
from PIL import Image
import base64
import io

load_dotenv()

apikey = os.getenv("WATSONX_API_KEY")
project_id = os.getenv("WATSONX_PROJECT_ID")
watsonx_ai_url = os.getenv("WATSONX_URL")

credentials = Credentials(
    url=f"{watsonx_ai_url}",
    api_key=f"{apikey}",
)
client = APIClient(credentials)

model = ModelInference(
    model_id="meta-llama/llama-3-2-11b-vision-instruct",
    api_client=client,
    project_id=f"{project_id}",
    params={"max_tokens": 3000},
)


def image_to_base64(image):
    """
    pillow 형식의 이미지를 가져와서 원하는 포맷으로 저장
    base64 인코딩 형식으로 이미지 리턴
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return image_base64


def recommend(message, history):

    print("history ", history)

    system_prompt = """
    당신은 전문 여행 플래너 AI 다.

    사용자가 업로드한 이미지의
    - 분위기
    - 감성
    - 색감
    - 스타일
    
    을 분석해서 여행지를 추천해줘

    반드시
    1. 이미지 분위기 분석
    2. 추천 여행지
    3. 추천 이유
    4. 추천 활동
    을 포함해줘

    항상 한국어로 답변해줘
    """

    messages = [
        {"role": "system", "content": system_prompt},
    ]
    # 사용하는 모델은 이미지를 한장만 해석 가능한 모델임
    # 이전 답변은 텍스트만 보냄
    for item in history:
        role = item["role"]
        content = item["content"]

        # assistant 응답 저장
        texts = []

        if isinstance(content, list):
            for c in content:

                if c.get("type") == "text":
                    texts.append(c.get("text", ""))
        elif isinstance(content, str):
            texts.append(content)

        messages.append({"role": role, "content": " ".join(texts)})

    # message : text, files
    text = message.get("text", "")
    files = message.get("files", "")

    if files:
        image = Image.open(files[0])

        # base64 인코딩 후
        base64_image = image_to_base64(image)
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                    {"type": "text", "text": text},
                ],
            }
        )
    else:
        messages.append({"role": "user", "content": text})

    # chat_stream()
    generated_response = model.chat_stream(messages=messages)

    full_response = ""
    for chunk in generated_response:
        if chunk["choices"]:
            full_response += chunk["choices"][0]["delta"].get("content", "")
            yield full_response


demo = gr.ChatInterface(
    fn=recommend,
    multimodal=True,
    title="🎈 AI 감성 여행 플래너",
    description="가고 싶은 여행지의 사진과 여행 스타일을 입력하면 AI가 여행일정을 추천해줍니다.",
)

demo.launch()
