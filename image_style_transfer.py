# 이미지 스타일 적용 변환

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import gradio as gr

# 이미지 전처리 함수
def preprocess_image(image, target_size=None):
    # 이미지를 NumPy 배열로 변환 및 정규화
    image = np.array(image).astype(np.float32) / 255.0
    if target_size:
        image = tf.image.resize(image, target_size)  # 이미지 크기 조정
    return tf.expand_dims(image, axis=0)  # 배치 차원 추가

# 스타일 변환 함수
def style_transfer(content_image, style_image):
    try:
        # TensorFlow Hub 모듈 로드
        hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

        # 이미지 전처리
        content_image = preprocess_image(content_image)
        style_image = preprocess_image(style_image, target_size=(256, 256))

        # 스타일 변환 수행
        outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
        stylized_image = outputs[0][0]

        # 결과 이미지 후처리
        stylized_image = (stylized_image * 255).numpy().astype(np.uint8)
        return Image.fromarray(stylized_image)

    except Exception as e:
        print(f"[오류 발생] {e}")
        return Image.new("RGB", (256, 256), "red")

# Gradio 인터페이스 설정
def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# 이미지 스타일 변환 웹")
        gr.Markdown("**왼쪽에는 변환할 이미지를, 오른쪽에는 변환하길 원하는 스타일의 이미지(예: 피카소, 모네 등)를 업로드하고 변환 버튼을 누르면 이미지가 변환됩니다.**")

        with gr.Row():
            content_image = gr.Image(label="변환할 이미지", type="pil")
            style_image = gr.Image(label="스타일 이미지(적용할 스타일)", type="pil")

        submit_button = gr.Button("스타일 변환")
        
        output_image = gr.Image(label="변환된 이미지", type="pil")


        # 버튼과 함수 연결
        submit_button.click(
            fn=style_transfer,
            inputs=[content_image, style_image],
            outputs=output_image
        )

    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()