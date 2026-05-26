import gradio as gr
from transformers import pipeline
import edge_tts
import asyncio

whisper = pipeline("automatic-speech-recognition", model="openai/whisper-base")
generator = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")


voice_txt, current_answer = "", ""


def change_txt(file):
    global voice_txt
    result = whisper(file, return_timestamps=True)
    voice_txt = result["text"]
    return voice_txt


def answer_question(question):
    global voice_txt, current_answer

    if not voice_txt:
        return "음성을 텍스트로 변환한 후 질문하세요"

    prompt = f"""
    본론 내용을 기반으로 사용자 질문에 대해서 대답을 해주세요
    본론: {voice_txt}

    사용자 질문: {question}

    답변: 
    """

    result = generator(
        prompt,
        max_new_tokens=50,
        return_full_text=False,
        do_sample=False,
        pad_token_id=generator.tokenizer.eos_token_id,
    )
    current_answer = result[0]["generated_text"].strip()
    return current_answer


async def text_to_voice(text):

    voice = "ko-KR-InJoonNeural"

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save("answer.mp3")


def make_voice():
    global current_answer
    if not current_answer:
        return None

    asyncio.run(text_to_voice(current_answer))

    return "answer.mp3"


with gr.Blocks(title="AI 음성 챗봇") as demo:
    gr.Markdown("## AI 음성 비서")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("오디오")
            file = gr.Audio(type="filepath")
            txt_btn = gr.Button("텍스트 변환")
        with gr.Column(scale=1):
            gr.Markdown("텍스트 변환")
            out = gr.Textbox(lines=5, interactive=False)
    with gr.Row():
        with gr.Column(scale=1):
            question_input = gr.Textbox(label="question")
            question_btn = gr.Button("질문하기")
        with gr.Column(scale=1):
            answer_output = gr.Textbox(label="answer")
            voice_btn = gr.Button("답변 음성 변환")
    with gr.Column():
        audio_output = gr.Audio(
            label="AI 음성 답변", type="filepath", interactive=False
        )

    txt_btn.click(fn=change_txt, inputs=file, outputs=out)
    question_btn.click(fn=answer_question, inputs=question_input, outputs=answer_output)
    voice_btn.click(fn=make_voice, outputs=audio_output)

demo.launch()
