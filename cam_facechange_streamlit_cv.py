# streamlit run cam_facechange_streamlit_cv.py

import streamlit as st
import cv2
import random
import numpy as np
import os
import time

# Haar Cascade 경로 설정
face_cascade = cv2.CascadeClassifier(os.path.join(os.getcwd(), "models", "haarcascade_frontalface_default.xml"))

# 얼굴 추적 및 합성 함수
def detect_and_swap_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    if len(faces) < 2:
        # 얼굴이 2개 미만일 경우 테두리만 그린 원본 반환
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame

    # 얼굴 영역 추출
    extracted_faces = []
    for (x, y, w, h) in faces:
        face_region = frame[y:y+h, x:x+w].copy()
        extracted_faces.append((face_region, (x, y, w, h)))

    # 얼굴 섞기
    random.shuffle(extracted_faces)

    # 얼굴 교체
    for i, (x, y, w, h) in enumerate(faces):
        face, (sx, sy, sw, sh) = extracted_faces[i]  # 교체할 얼굴과 원본 위치
        resized_face = cv2.resize(face, (w, h))  # 얼굴 크기 조정

        # 얼굴 합성을 위한 마스크 생성
        mask = 255 * np.ones(resized_face.shape, resized_face.dtype)

        # 얼굴 합성
        center = (x + w // 2, y + h // 2)
        frame = cv2.seamlessClone(resized_face, frame, mask, center, cv2.NORMAL_CLONE)

        # 얼굴 테두리 표시
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame

# Streamlit
st.title("실시간 얼굴 바꾸기")
st.text("웹캠을 켜고 두 사람 이상 들어오면 얼굴 인식 후 서로 얼굴을 바꿔줍니다.")

# 상태 변수 초기화
if "streaming" not in st.session_state:
    st.session_state.streaming = False

# 버튼 UI
col1, _, col2 = st.columns([0.4, 0.07, 2])  # 가운데 빈 열로 간격 조절
with col1:
    start_button = st.button("웹캠 시작")  
with col2:
    stop_button = st.button("웹캠 끄기")  


# 웹캠 스트리밍 제어
if start_button:
    st.session_state.streaming = True

if stop_button:
    st.session_state.streaming = False

# 스트리밍 실행
if st.session_state.streaming:
    cap = cv2.VideoCapture(0)

    try:
        if not cap.isOpened():
            st.error("웹캠을 찾을 수 없습니다. 웹캠이 연결되었는지 확인하세요.")
            st.session_state.streaming = False
        else:
            frame_placeholder = st.empty()
            while st.session_state.streaming:
                ret, frame = cap.read()
                if not ret:
                    st.error("프레임을 읽을 수 없습니다. 스트리밍을 다시 시작하세요.")
                    st.session_state.streaming = False
                    break

                # 얼굴 감지 및 교체
                swapped_frame = detect_and_swap_faces(frame)

                # 결과 프레임 표시
                frame_placeholder.image(cv2.cvtColor(swapped_frame, cv2.COLOR_BGR2RGB), channels="RGB")
                time.sleep(0.03)  # UI 반응 속도를 위한 딜레이
    finally:
        cap.release()

