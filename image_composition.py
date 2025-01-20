import gradio as gr
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub

# TensorFlow Hub의 깊이 예측 모델 로드
MODEL_URL = "https://tfhub.dev/intel/midas/v2_1_small/1"
try:
    depth_model = hub.load(MODEL_URL)
    print("모델 로드 성공")
except Exception as e:
    depth_model = None
    print(f"모델 로드 실패: {e}")
    exit()

def generate_depth_map(image):
    # 이미지를 모델 입력 크기에 맞게 조정 (256x256)
    input_tensor = tf.image.resize(image, (256, 256))
    input_tensor = tf.cast(input_tensor, tf.float32) / 255.0  # [0, 1]로 정규화
    input_tensor = tf.expand_dims(input_tensor, axis=0)  # 배치 차원 추가

    # 모델 호출
    output = depth_model(input_tensor)

    # 깊이 맵 추출
    depth_map = output["default"].numpy()[0]

    # 깊이 맵 정규화
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return depth_map

def apply_3d_effect(image, depth_map, x_offset, y_offset):
    # 이미지와 깊이 맵의 크기 동기화
    h, w = image.shape[:2]
    depth_map = cv2.resize(depth_map, (w, h))

    # 왜곡 효과 적용
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x_new = np.clip(x + depth_map * (x_offset / 100), 0, w - 1).astype(np.float32)
    y_new = np.clip(y + depth_map * (y_offset / 100), 0, h - 1).astype(np.float32)
    remapped = cv2.remap(image, x_new, y_new, interpolation=cv2.INTER_LINEAR)

    return remapped

def process_image(image, x_offset, y_offset):
    try:
        # 이미지를 NumPy 배열로 변환
        image_np = np.array(image)

        # 깊이 맵 생성
        depth_map = generate_depth_map(image_np)

        # 3D 효과 적용
        output_image = apply_3d_effect(image_np, depth_map, x_offset, y_offset)

        return Image.fromarray(output_image), Image.fromarray(depth_map)
    except Exception as e:
        print(f"오류 발생: {e}")
        return None, None

def setup_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# 3D 이미지 효과\n이미지를 업로드하고 가상 카메라 뷰포인트를 조정하세요.")

        with gr.Row():
            input_image = gr.Image(label="업로드 이미지", type="pil")
            output_image = gr.Image(label="3D 효과 적용 이미지")
            depth_map_output = gr.Image(label="생성된 깊이 맵")

        x_offset = gr.Slider(label="카메라 X 이동", minimum=-50, maximum=50, step=1, value=0)
        y_offset = gr.Slider(label="카메라 Y 이동", minimum=-50, maximum=50, step=1, value=0)

        inputs = [input_image, x_offset, y_offset]
        outputs = [output_image, depth_map_output]

        def update(*args):
            return process_image(*args)

        # 실시간 업데이트 연결
        for control in inputs:
            control.change(fn=update, inputs=inputs, outputs=outputs, queue=False)

    return demo

if __name__ == "__main__":
    demo = setup_ui()
    demo.launch()
