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
   url = f"{watsonx_ai_url}",
   api_key = f"{apiKey}",
)
client = APIClient(credentials)

model = ModelInference(
    model_id="ibm/granite-4-h-small",
    api_client=client,
    project_id=f"{project_id}",
    params = {
        "max_tokens": 1000,
        "temperature": 0.55,
    }
)


def ad_text(name, brand_name, strength, tone, keyword, value):

  
    instructions = f"""
    광고 문구 프로그램 입니다.
    당신은 광고 업계에서 유명한 카피라이터입니다. 
    광고 문구 프로그램을 만들어서 제공하려고 합니다.
    당신의 임무는 주어진 조건을 이용해 창의적인 광고 문구를 작성하는 것입니다.

    사용자에게 6가지 입력값을 받았습니다.
    제품명: {name}
    브랜드명: {brand_name}
    제품특징: {strength}
    톤앤매너: {tone} 
    필수 포함 키워드: {keyword}
    브랜드 핵심 가치: {value}

    사용자에게 받은 정보를 가지고
    제품을 홍보하는 광고 문구를 만들어주세요 

    광고 문구는 5가지만 만들어주세요
    - 광고 문구를 만들 때 톤앤매너를 신경써주세요
    - 1,2,3,4,5 번호 줘서 나열해주시고, 문구를 만들게 된 배경을 같이 작성해주세요
    - 이 템플릿으로 답변을 작성해주세요. 숫자, 문구, 작성 배경은 다 한줄씩 띄어서 
      1 
      - 문구: 
      - 작성 배경: 
    - 1번 문구는 제품의 특징이 잘 드러나도록 작성해주세요
    - 2번 문구는 브랜드의 핵심 가치를 잘 드러나도록 작성해주세요
"""
    
    user_prompt = f"""
    아래 내용을 참고해서 1-2줄 짜리 광고 문구 5개 작성해줘.
    - 제품명: {name}
    - 브랜드명: {brand_name}
    - 제품특징: {strength}
    - 톤앤매너: {tone} 
    - 필수 포함 키워드: {keyword}
    - 브랜드 핵심 가치: {value}

"""
    messages = [
        {"role": "system", "content":instructions}, 
        # {"role": "user", "content":user_prompt }
    ]

    generated_response = model.chat(messages=messages)
    return generated_response['choices'][0]['message']['content']

demo = gr.Interface(
    fn=ad_text,
    inputs=[
        gr.Text(label="제품명"), 
        gr.Textbox(label="브랜드명"), 
        gr.Textbox(label="제품특징"), 
        gr.Textbox(label="톤앤매너"),  
        gr.Textbox(label="필수 포함 키워드"), 
        gr.Textbox(label="브랜드 핵심 가치")
    ],
    outputs=[gr.Markdown()],
    title="🙌 광고 문구 프로그램",
    description="텍스트 작성 시 AI가 광고 문구를 작성해 드립니다"
)

demo.launch()