import gradio as gr

def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)

# Interface: 빠르게 만들 때
demo = gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text"], # greet 함수 return 값
    api_name="predict"
)

demo.launch()