# streamlit run cam_facemask_streamlit.py
# 그라디오로 구현하고 싶었는데 실패


import streamlit as st
import cv2 as cv
import mediapipe as mp
import numpy as np
from PIL import Image
import time

# Streamlit 설정
st.title("실시간 증강현실 마스크")
st.text("카메라 정면을 바라봐주세요.")
st.text("웹캠에서 얼굴을 감지하고 마스크를 씌워줍니다.")

# Mediapipe 설정
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# 마스크 이미지 로드 함수
def load_mask(mask_name):
    mask_path = f'./images/mask/{mask_name}'
    mask = cv.imread(mask_path, cv.IMREAD_UNCHANGED)
    if mask is None:
        st.error(f"{mask_name} 이미지를 로드할 수 없습니다.")
        st.stop()

    if mask.shape[-1] != 4:  # 알파 채널 확인
        st.error(f"{mask_name} 이미지에 알파 채널이 포함되어 있지 않습니다.")
        st.stop()

    return mask

# 기본 마스크 및 추가 마스크 목록
default_mask = "mask.png"
mask_options = [
    "mask.png",  # 기본 마스크
    "mask_animal.png",
    "mask_pekingopera.png",
    "mask_butterfly.png",
    "mask_v.png",
    "mask_spyder.png"
]

selected_mask_name = st.selectbox("마스크 선택", mask_options, index=0)
dice = load_mask(selected_mask_name)

# 기본 마스크 크기 참조
# default_dice = load_mask(default_mask)
# default_width, default_height = default_dice.shape[1], default_dice.shape[0]

# 전역 변수로 웹캠 상태 관리
cap = None

# 마스크 적용 함수
def apply_mask(frame):
    # OpenCV는 BGR, Streamlit은 RGB이므로 변환
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Mediapipe 얼굴 탐지 실행
    res = face_detection.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    if res.detections:
        for det in res.detections:
            bboxC = det.location_data.relative_bounding_box
            x1, y1 = int(bboxC.xmin * frame.shape[1]), int(bboxC.ymin * frame.shape[0])
            x2, y2 = int((bboxC.xmin + bboxC.width) * frame.shape[1]), int((bboxC.ymin + bboxC.height) * frame.shape[0])

            # 얼굴 크기에 맞게 마스크 조정 (1.8배 확대)
            face_width, face_height = int((x2 - x1) * 1.7), int((y2 - y1) * 1.8)
            resized_dice = cv.resize(dice, (face_width, face_height))

            # 중심점 계산 및 조정
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            offset = int(face_height * 0.085)  # 얼굴 높이의 8.5%만큼 위로 이동
            center_y -= offset

            x1_new, y1_new = center_x - face_width // 2, center_y - face_height // 2
            x2_new, y2_new = x1_new + face_width, y1_new + face_height

            # 투명도 계산 및 이미지 합성
            alpha = resized_dice[:, :, 3:] / 255
            if x1_new > 0 and y1_new > 0 and x2_new < frame.shape[1] and y2_new < frame.shape[0]:
                frame[y1_new:y2_new, x1_new:x2_new] = (
                    frame[y1_new:y2_new, x1_new:x2_new] * (1 - alpha) +
                    resized_dice[:, :, :3] * alpha
                )

    return frame

# 웹캠 스트리밍 루프
def webcam_stream(frame_window):
    global cap
    cap = cv.VideoCapture(0)  # 웹캠 열기
    if not cap.isOpened():
        st.error("카메라를 열 수 없습니다.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("웹캠에서 프레임을 읽을 수 없습니다.")
            break

        # 마스크 적용
        frame = apply_mask(frame)

        # Streamlit에서 이미지 표시
        frame_window.image(frame, channels="RGB")

        # 루프를 잠시 멈춰 CPU 사용률 감소
        time.sleep(0.03)

    cap.release()

# UI: 웹캠 제어 버튼
frame_window = st.image([])  # 화면에 출력할 스트림 공간
col1, col2 = st.columns([0.2, 0.8])  # 버튼 간 간격 조정
with col1:
    st.button("웹캠 켜기", on_click=lambda: webcam_stream(frame_window))
with col2:
    st.button("웹캠 끄기", on_click=lambda: cap.release() if cap and cap.isOpened() else None)

