# 이미지 변환

import gradio as gr
import cv2
import numpy as np
from PIL import Image

def process_image(image, grayscale, rotation, scale, red_intensity, green_intensity, blue_intensity, brightness, sharpen):
    image = np.array(image)

    # 그레이스케일 변환
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # 이미지 회전
    if rotation != 0:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
        image = cv2.warpAffine(image, matrix, (w, h))

    # 크기 조정
    if scale != 100:
        new_width = int(image.shape[1] * scale / 100)
        new_height = int(image.shape[0] * scale / 100)
        image = cv2.resize(image, (new_width, new_height))

    # 색상 조정
    image[:, :, 0] = np.clip(image[:, :, 0] * (red_intensity / 100), 0, 255)  # Red
    image[:, :, 1] = np.clip(image[:, :, 1] * (green_intensity / 100), 0, 255)  # Green
    image[:, :, 2] = np.clip(image[:, :, 2] * (blue_intensity / 100), 0, 255)  # Blue

    # 밝기 조정
    if brightness != 0:
        image = cv2.convertScaleAbs(image, alpha=1, beta=brightness)

    # 샤프닝
    if sharpen != 0:
        kernel = np.array([[0, -1, 0], [-1, 5 + sharpen / 10, -1], [0, -1, 0]])
        image = cv2.filter2D(image, -1, kernel)

    return Image.fromarray(image)

def setup_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# 이미지 변환 웹\n이미지를 업로드하면 원하는대로 변환할 수 있습니다.")

        with gr.Row():
            input_image = gr.Image(label="업로드 이미지", type="pil")
            output_image = gr.Image(label="변환된 이미지")

        grayscale = gr.Checkbox(label="그레이스케일 변환")

        with gr.Row():
            rotation_buttons = gr.Radio(
                label="이미지 회전 (각도)",  
                choices=[0, 90, 180, 270, 360],
                type="value",
                value=0
            )

        scale = gr.Slider(label="크기 조정", minimum=50, maximum=200, step=5, value=100)
        red_intensity = gr.Slider(label="Red 강도", minimum=0, maximum=200, step=5, value=100)
        green_intensity = gr.Slider(label="Green 강도", minimum=0, maximum=200, step=5, value=100)
        blue_intensity = gr.Slider(label="Blue 강도", minimum=0, maximum=200, step=5, value=100)
        brightness = gr.Slider(label="밝기 조정", minimum=-100, maximum=100, step=3, value=0)
        sharpen = gr.Slider(label="샤프닝 강도", minimum=0, maximum=50, step=3, value=0)

        # 컨트롤과 이미지 업데이트 연결
        def update_controls(*inputs):
            if inputs[0] is None:  
                return None
            return process_image(*inputs)

        controls = [input_image, grayscale, rotation_buttons, scale, red_intensity, green_intensity, blue_intensity, brightness, sharpen]
        for control in controls[1:]:  
            control.change(
                update_controls,
                inputs=controls,
                outputs=[output_image]
            )

    return demo

if __name__ == "__main__":
    demo = setup_ui()
    demo.launch()
