# pip install 'transformers[sentencepiece]==4.41.2' gradio python-dotenv torch

import gradio as gr
from transformers import pipeline

# 형태 확인
# message {'text': '텍스트 이미지 ', 'files': ['~\\4b.jpg']}

captioner = pipeline("image-to-text")  # 이미지 캡셔닝
generator = pipeline(
    "text-generation", model="Qwen/Qwen2.5-1.5B-Instruct"
)  # 텍스트 생성

current_caption = ""


def chat(message, history):

    # 전역변수 사용할 수 있도록
    global current_caption

    # message에서 이미지 분리
    text = message.get("text")
    files = message.get("files", [])
    image = files[0] if files else None

    if image:
        result = captioner(image)
        caption_result = result[0]["generated_text"]
        current_caption = caption_result

        prompt = f"""
        이미지 설명: 
        {caption_result}
        
        사용자 질문: 
        {text}
        """
        return prompt
    else:

        if not current_caption:
            return f"""
            사용자 질문:
            {text}
            """

        prompt = """
            당신은 이미지 분석 AI 입니다.
            다음 이미지 설명을 참고해서 사용자의 질문에 한 문장으로 답변하세요.

            이미지 설명:
            {current_caption}

            사용자 질문:
            {text}

            답변:
        """
        result = generator(
            prompt,
            max_new_tokens=50,
            return_full_text=False,
            pad_token_id=generator.tokenizer.eos_token_id,
        )

        print("text result", result)

        response = result[0]["generated_text"]
        answer = response.split("\n")[0].strip()

        return answer


demo = gr.ChatInterface(
    fn=chat,
    textbox=gr.MultimodalTextbox(
        file_types=["image"],
        file_count="single",
    ),
    title="멀티 모달 AI",
    description="이미지를 업로드하면 이미지에 대한 설명을 생성하는 챗 봇",
)
demo.launch()
