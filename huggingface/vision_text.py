import gradio as gr
from transformers import pipeline

# 형태 확인
# text만
# message {'text': '텍스트만', 'files': []}

# file만
# message {'text': '', 'files': ['~\\4b.jpg']}

# text + file
# message {'text': '텍스트 이미지 ', 'files': ['~\\4b.jpg']}


# 이미지 캡셔닝
captioner = pipeline("image-to-text")


def chat(message, history):
    print("message", message)
    print("history", history)

    # message에서 이미지 분리
    text = message.get("text")
    files = message.get("files", [])
    image = files[0] if files else None

    if image:
        result = captioner(image)
        return f"이미지 캡션: {result[0]['generated_text']}"
    else:
        return text


demo = gr.ChatInterface(
    fn=chat,
    textbox=gr.MultimodalTextbox(
        file_types=["image"],
        file_count="single",
    ),
    title="멀티 모달 AI",
    description="이미지를 업로드하면 이미지에 대한 설명을 생성하는 챗 봇",
)
demo.launch(debug=True)
