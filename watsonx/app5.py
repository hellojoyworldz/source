import gradio as gr
# pillow(파이썬 이미지 처리 라이브러리) 사용법 알아야 함
from PIL import ImageEnhance

def process_image(editor_value):
    '''
    composite: 원본 그림 위에 레이어 반영한 최종 이미지
    '''

    image = editor_value['composite']
    enhancer = ImageEnhance.Brightness(image)
    result = enhancer.enhance(1.5)

    return result

interface = gr.Interface(process_image,gr.ImageEditor("pil"), gr.Image() )
interface.launch()