import gradio as gr
from PIL import Image
import rembg  # 배경 제거 라이브러리
import numpy as np  
import os  

BACKGROUND_DIR = "./images/background"  
BACKGROUND_IMAGES = {
    "해변 배경": os.path.join(BACKGROUND_DIR, "beach.jpg"),
    "우주 배경": os.path.join(BACKGROUND_DIR, "space.png"),  # PNG 파일
    "도시 배경": os.path.join(BACKGROUND_DIR, "city.jpg"),
    "숲 배경": os.path.join(BACKGROUND_DIR, "forest.jpg"),
}

# 배경 제거 함수
def remove_background(image):
    output = rembg.remove(image)
    return output

# 이미지와 배경 합성 함수
def composite_image(image, background):
    # NumPy 배열을 PIL.Image로 변환 (알파 채널 처리)
    if isinstance(image, np.ndarray):
        if image.shape[2] == 4:  # 알파 채널이 있는 경우
            image = Image.fromarray(image, mode="RGBA")
        else:  # 알파 채널이 없는 경우
            image = Image.fromarray(image, mode="RGB")
    
    # 배경 처리
    if isinstance(background, str):  # 배경이 문자열일 경우
        if background == "투명 배경":
            return image
        elif background in BACKGROUND_IMAGES:
            bg_image = Image.open(BACKGROUND_IMAGES[background])
        else:
            raise ValueError(f"알 수 없는 배경: {background}")
    elif isinstance(background, Image.Image):  # 배경이 PIL.Image일 경우
        bg_image = background
    else:
        raise TypeError("background는 문자열 또는 PIL.Image 객체여야 합니다.")

    # 배경 이미지를 원본 이미지 크기로 리사이즈
    bg_image = bg_image.resize(image.size)

    # 배경 위에 원본 이미지 합성 (중앙 배치)
    result = bg_image.copy()
    if image.mode == "RGBA":  # 알파 채널이 있는 경우
        result.paste(image, (0, 0), image)
    else:  # 알파 채널이 없는 경우
        result.paste(image, (0, 0))
    
    return result



# 전체 프로세스 함수
def process_image(image, background):
    # 배경 제거
    removed_bg = remove_background(image)
    # 배경 합성
    result = composite_image(removed_bg, background)
    return result

# 다양한 배경에 적용하는 함수
def fun_backgrounds(image):
    results = []
    for bg_name, bg_path in BACKGROUND_IMAGES.items():
        try:
            bg_image = Image.open(bg_path)
            # 배경 제거된 이미지가 NumPy 배열인지 확인 후 변환
            if isinstance(image, np.ndarray):
                if image.shape[2] == 4:  # 알파 채널이 있는 경우
                    image_pil = Image.fromarray(image, mode="RGBA")
                else:  # 알파 채널이 없는 경우
                    image_pil = Image.fromarray(image, mode="RGB")
            else:
                image_pil = image
            result = composite_image(image_pil, bg_image)
            results.append(result)
        except FileNotFoundError:
            print(f"경고: 배경 이미지 파일을 찾을 수 없습니다 - {bg_path}")
    return results

# Gradio 인터페이스
with gr.Blocks() as demo:
    gr.Markdown("## 이미지 배경 제거 및 원하는 배경으로 합성하기")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="이미지 업로드")
            background_choice = gr.Dropdown(
                choices=["투명 배경", "해변 배경", "우주 배경", "도시 배경", "숲 배경", "배경 이미지 업로드"],
                label="배경 선택"
            )
            custom_background = gr.Image(label="합성할 배경 선택", visible=False)
        
        with gr.Column():
            output_image = gr.Image(label="배경 합성 결과")
    
    # 배경 선택 변경 시 사용자 지정 배경 업로드 필드 표시
    background_choice.change(
        fn=lambda x: gr.update(visible=x == "사용자 지정 배경"),
        inputs=background_choice,
        outputs=custom_background
    )
    
    # 이미지 및 배경 선택 변경 시 결과 이미지 업데이트
    def update_image(image, background, custom_bg):
        # 입력 이미지가 없을 경우
        if image is None:
            return None  

        # 사용자 지정 배경 선택 시
        if background == "사용자 지정 배경":
            if custom_bg is None:
                return image
            else:
                # 사용자 지정 배경이 NumPy 배열인 경우 PIL.Image로 변환
                if isinstance(custom_bg, np.ndarray):
                    custom_bg = Image.fromarray(custom_bg)

                # custom_bg가 올바른 PIL.Image 객체인지 확인
                if not isinstance(custom_bg, Image.Image):
                    raise ValueError("업로드한 이미지가 올바르지 않습니다.")
                
                # 사용자 지정 배경과 합성
                return process_image(image, custom_bg)
        else:
            # 기본 배경과 합성
            return process_image(image, background)


# Gradio 인터페이스
with gr.Blocks() as demo:
    gr.Markdown("## 이미지 배경 제거 & 원하는 배경 합성")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="이미지 업로드")
            background_choice = gr.Dropdown(
                choices=["투명 배경", "해변 배경", "우주 배경", "도시 배경", "숲 배경", "사용자 지정 배경"],
                label="배경 선택"
            )
            custom_background = gr.Image(label="사용자 지정 배경 업로드", visible=False)
        
        with gr.Column():
            output_image = gr.Image(label="결과 이미지")
    
    # 배경 선택 변경 시 사용자 지정 배경 업로드 필드 표시
    background_choice.change(
        fn=lambda x: gr.update(visible=x == "사용자 지정 배경"),
        inputs=background_choice,
        outputs=custom_background
    )
    
    # 이미지 및 배경 선택 변경 시 결과 이미지 업데이트
    input_image.change(
        fn=update_image,
        inputs=[input_image, background_choice, custom_background],
        outputs=output_image
    )

    # 사용자 지정 배경 업로드 시 업데이트
    custom_background.change(
        fn=update_image,
        inputs=[input_image, background_choice, custom_background],
        outputs=output_image
    )

    background_choice.change(
        fn=update_image,
        inputs=[input_image, background_choice, custom_background],
        outputs=output_image
    )

demo.launch()