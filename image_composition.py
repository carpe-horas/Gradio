import gradio as gr
import cv2
import numpy as np
from PIL import Image


def blend_images(image1, image2, image3, image4, alpha1, alpha2, alpha3, alpha4):
    # 이미지를 NumPy 배열로 변환 및 크기 통일
    image1 = np.array(image1.resize((500, 500))) if image1 else np.zeros((500, 500, 3), dtype=np.uint8)
    image2 = np.array(image2.resize((500, 500))) if image2 else np.zeros((500, 500, 3), dtype=np.uint8)
    image3 = np.array(image3.resize((500, 500))) if image3 else np.zeros((500, 500, 3), dtype=np.uint8)
    image4 = np.array(image4.resize((500, 500))) if image4 else np.zeros((500, 500, 3), dtype=np.uint8)

    # 합성 초기값 설정 (빈 이미지로 시작)
    blended = np.zeros_like(image1, dtype=np.float32)

    # 알파 값에 따라 각각의 이미지를 합성
    blended += image1 * (alpha1 / 100)
    blended += image2 * (alpha2 / 100)
    blended += image3 * (alpha3 / 100)
    blended += image4 * (alpha4 / 100)

    # 값의 범위를 0~255로 제한하고 uint8 형식으로 변환
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)


def setup_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# 이미지 합성 웹\n4개의 이미지를 업로드하고 각각의 투명도를 조절하여 합성하세요.")

        with gr.Row():
            # 이미지 업로드
            image1 = gr.Image(label="이미지 1 업로드", type="pil")
            image2 = gr.Image(label="이미지 2 업로드", type="pil")
            image3 = gr.Image(label="이미지 3 업로드", type="pil")
            image4 = gr.Image(label="이미지 4 업로드", type="pil")

        # 트랙바로 각 이미지의 투명도 조정
        alpha1 = gr.Slider(label="이미지 1", minimum=0, maximum=100, step=1, value=45)
        alpha2 = gr.Slider(label="이미지 2", minimum=0, maximum=100, step=1, value=45)
        alpha3 = gr.Slider(label="이미지 3", minimum=0, maximum=100, step=1, value=45)
        alpha4 = gr.Slider(label="이미지 4", minimum=0, maximum=100, step=1, value=45)  # 수정된 부분

        # 합성된 이미지 출력
        output_image = gr.Image(label="합성된 이미지")

        # 실시간 업데이트 연결
        def update_output(image1, image2, image3, image4, alpha1, alpha2, alpha3, alpha4):
            return blend_images(image1, image2, image3, image4, alpha1, alpha2, alpha3, alpha4)

        inputs = [image1, image2, image3, image4, alpha1, alpha2, alpha3, alpha4]
        for control in inputs:
            control.change(fn=update_output, inputs=inputs, outputs=output_image)

    return demo


if __name__ == "__main__":
    demo = setup_ui()
    demo.launch()