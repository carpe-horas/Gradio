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









# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import gradio as gr
# from tensorflow.keras.applications.vgg19 import preprocess_input

# # 이미지 로드 및 전처리 함수
# def load_image(img):
#     print("[DEBUG] load_image 함수 호출")
#     if img is None:
#         raise ValueError("업로드된 이미지가 없습니다.")
#     print("[DEBUG] 업로드된 이미지 타입:", type(img))

#     # 이미지를 numpy 배열로 변환
#     img = np.array(img)
#     print("[DEBUG] numpy 배열 변환 완료:", img.shape)

#     # (H, W, C) 형식인지 확인
#     if len(img.shape) != 3 or img.shape[2] != 3:
#         raise ValueError("이미지가 잘못된 형식입니다. RGB 이미지가 필요합니다.")
    
#     # 이미지 크기 조정
#     img = Image.fromarray(img)
#     img = img.resize((224, 224))  # VGG19 모델의 입력 크기
#     print("[DEBUG] 이미지 크기 조정 완료:", img.size)

#     img = np.array(img, dtype=np.float32)
#     print("[DEBUG] numpy 배열로 변환 완료:", img.shape)

#     img = np.expand_dims(img, axis=0)  # (H, W, C) -> (1, H, W, C)
#     print("[DEBUG] 차원 추가 완료:", img.shape)

#     img = preprocess_input(img)  # VGG19에 맞는 전처리 수행
#     print("[DEBUG] 전처리 완료:", img.shape)

#     return tf.Variable(img)

# # VGG19 모델 로드
# def load_vgg19_model():
#     print("[DEBUG] VGG19 모델 로드")
#     vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
#     vgg.trainable = False
#     outputs = [vgg.get_layer(name).output for name in 
#                ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']]
#     model = tf.keras.Model([vgg.input], outputs)
#     print("[DEBUG] VGG19 모델 출력 계층 정의 완료")
#     return model

# # 스타일 손실 계산
# def compute_style_loss(style_outputs, target_outputs):
#     loss = 0
#     for style, target in zip(style_outputs, target_outputs):
#         loss += tf.reduce_mean(tf.square(style - target))
#     return loss

# # 콘텐츠 손실 계산
# def compute_content_loss(content_output, target_output):
#     return tf.reduce_mean(tf.square(content_output - target_output))

# # Neural Style Transfer 함수
# def neural_style_transfer(content_img, style_img, num_iterations=100, style_weight=1e-2, content_weight=1e4):
#     print("[DEBUG] 스타일 변환 시작")
#     content_img = load_image(content_img)
#     style_img = load_image(style_img)
#     print("[DEBUG] 입력 이미지 로드 완료")

#     vgg = load_vgg19_model()
#     print("[DEBUG] VGG19 모델 로드 완료")

#     # 스타일 및 콘텐츠 출력 계산
#     style_outputs = vgg(style_img)
#     content_output = vgg(content_img)[-1]
#     print("[DEBUG] 스타일 및 콘텐츠 출력 계산 완료")

#     generated_img = tf.Variable(content_img)
#     optimizer = tf.optimizers.Adam(learning_rate=0.01)

#     # 스타일 변환 반복
#     for i in range(num_iterations):
#         with tf.GradientTape() as tape:
#             generated_outputs = vgg(generated_img)
#             style_loss = compute_style_loss(generated_outputs[:-1], style_outputs[:-1])
#             content_loss = compute_content_loss(generated_outputs[-1], content_output)
#             total_loss = style_weight * style_loss + content_weight * content_loss

#         grads = tape.gradient(total_loss, generated_img)
#         optimizer.apply_gradients([(grads, generated_img)])

#         generated_img.assign(tf.clip_by_value(generated_img, -127.5, 127.5))

#         if i % 10 == 0:
#             print(f"[DEBUG] Iteration {i}: Total Loss: {total_loss.numpy()}")

#     output_img = generated_img.numpy().squeeze()
#     output_img = np.clip(output_img + 127.5, 0, 255)  # 픽셀 값 제한
#     return output_img.astype("uint8")

# # Gradio 인터페이스
# def gradio_interface(content_img, style_img):
#     try:
#         print("[DEBUG] Gradio 인터페이스 호출")
#         output_img = neural_style_transfer(content_img, style_img, num_iterations=50)
#         print("[DEBUG] 스타일 변환 완료, 이미지 반환")
#         return Image.fromarray(output_img)
#     except Exception as e:
#         print(f"[오류 발생] {e}")
#         return Image.new("RGB", (224, 224), "red")  # 오류 시 기본 이미지 반환

# # Gradio 설정
# iface = gr.Interface(
#     fn=gradio_interface,
#     inputs=["image", "image"],
#     outputs="image",
#     live=True,
#     allow_flagging="never"
# )

# if __name__ == "__main__":
#     print("[DEBUG] Gradio 애플리케이션 시작")
#     iface.launch()


