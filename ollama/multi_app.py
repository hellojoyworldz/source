import gradio as gr
from image_analyzer import process_documents, build_rag_chain

vectorstore = None
rag_chain = None


def upload_and_process(files):
    global vectorstore, rag_chain

    if not files:
        vectorstore = None
        rag_chain = None
        return "업로드된 파일이 없습니다."

    paths = [f.name for f in files]
    vectorstore = process_documents(files)
    rag_chain = build_rag_chain(vectorstore)

    return f"{len(paths)}개 문서 처리 완료. 질문하세요."


def answer_question(question):
    global rag_chain

    if rag_chain is None:
        return "먼저 문서를 업로드하고 분석을 시작하세요."
    if not question:
        return "질문을 입력하세요."

    return rag_chain.invoke(question)


with gr.Blocks(title="이미지 개반 문서 분석 시스템") as app:
    gr.Markdown("## 이미지 기반 문서 분석 시스템")
    gr.Markdown("스캔 문서, 영수증, 게약서 이미지를 업로드 하면 질문할 수 있습니다.")

    with gr.Column():
        file_input = gr.File(
            label="문서 이미지 업로드",
            file_types=[".jpg", ".jpeg", ".png", ".webp"],
            file_count="multiple",
        )
        status_bar = gr.Textbox(label="처리 상태", lines=3)
        upload_btn = gr.Button("문서 분석 시작", variant="primary")
        upload_btn.click(
            fn=upload_and_process,
            inputs=[file_input],
            outputs=[status_bar],
        )
    with gr.Column():
        gr.Markdown("---")
        question = gr.Textbox(
            label="질문 입력", placeholder="문서에서 찾고 싶은 내용을 입력하세요"
        )
        answer_box = gr.Textbox(label="답변", lines=8)
        ask_button = gr.Button("질문하기", variant="primary")
        ask_button.click(fn=answer_question, inputs=[question], outputs=[answer_box])

app.launch()
