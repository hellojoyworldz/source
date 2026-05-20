import gradio as gr
from transformers import pipeline
import re

english_classifier = pipeline("sentiment-analysis")
korean_classifier = pipeline(
    "sentiment-analysis", model="WhitePeak/bert-base-cased-Korean-sentiment"
)


def is_korean(text):
    korean = re.search(r"가-힣", text)
    return korean is not None


def predict_sentiment(text):
    # 한국말인지 확인하기

    if is_korean(text):
        result = korean_classifier(text)
    else:
        result = english_classifier(text)

    label = result[0]["label"]
    score = result[0]["label"]
    return f"감정: {label}\n확률: {score:.4f}"


demo = gr.Interface(
    fn=predict_sentiment,
    title="AI 감정분석 웹 앱",
    description="Hugging Face Transformer 모델 기반 감정 분석",
    inputs=[gr.Text(lines=3, placeholder="문장입력")],
    outputs=[gr.Text(lines=3, label="sentiment")],
)

demo.launch()
