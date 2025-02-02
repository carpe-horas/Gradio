# streamlit run cam_face_emotion_stremlit.py


import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

# 모델 로드 (사전 학습된 감정 분석 모델)
emotion_model_path = "./models/emotion_model_tf2.h5"
model = load_model(emotion_model_path)

# OpenCV 얼굴 감지기 로드 (경로 수정)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 감정 리스트 (FER2013 기준)
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# 감정별 배경 색상 설정
emotion_colors = {
    "Happy": "#90EE90",
    "Angry": "#FF4500",
    "Sad": "#4682B4",
    "Neutral": "#D3D3D3",
    "Surprise": "#FFD700",
    "Fear": "#800080",
    "Disgust": "#228B22"
}

# Streamlit UI
st.title("🎭 실시간 얼굴 감정 분석")
st.write("웹캠을 통해 얼굴 감정을 분석하고 화면에 현재 감정을 나타냅니다.")

# 웹캠 상태 관리 변수 초기화
if "webcam_active" not in st.session_state:
    st.session_state["webcam_active"] = False
if "current_emotion" not in st.session_state:
    st.session_state["current_emotion"] = None  #  처음엔 감정 상태 없음

# 감정 상태 실시간 업데이트 공간 확보
emotion_display = st.empty()

if st.session_state["current_emotion"] is None:
    emotion_display.markdown("### 현재 감정 상태: -")  # 웹캠 실행 전에는 "-"로 표시
else:
    emotion_display.markdown(f"### 현재 감정 상태: **{st.session_state['current_emotion']}**")


col1, col2 = st.columns([0.2, 0.8])

with col1:
    if st.button("📷 웹캠 실행"):
        st.session_state["webcam_active"] = True

with col2:
    if st.button("🛑 웹캠 끄기"):
        # 모든 상태 초기화 후 UI 새로고침
        st.session_state["webcam_active"] = False
        st.session_state["current_emotion"] = "Neutral"
        st.rerun()  # UI 전체를 새로고침하여 즉시 종료

# 감정별 배경 색상 변경 함수
def change_background_color(emotion):
    color = emotion_colors.get(emotion, "#FFFFFF")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {color};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# 웹캠 프레임 처리 함수
def detect_emotion(frame):
    if not st.session_state["webcam_active"]:  # 웹캠이 종료된 경우 즉시 반환
        return frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        if not st.session_state["webcam_active"]:  # 예측 전에 다시 한 번 상태 확인
            return frame

        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (64, 64))
        roi_gray = np.expand_dims(roi_gray, axis=-1)
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = roi_gray / 255.0

        if not st.session_state["webcam_active"]:  # 예측 전에 다시 한 번 확인
            return frame

        try:
            predictions = model.predict(roi_gray)  # 웹캠이 꺼지면 예측 수행 X
            max_index = np.argmax(predictions)
            emotion = emotion_labels[max_index]
        except Exception as e:
            st.warning("웹캠이 꺼지는 도중 감정 예측을 중단했습니다.")
            return frame  # 예측 중 에러 발생 시 안전 종료

        # 감정 상태 업데이트
        st.session_state["current_emotion"] = emotion
        emotion_display.markdown(f"### 현재 감정 상태: **{st.session_state['current_emotion']}**")

        # 배경 색상 변경 적용
        change_background_color(emotion)

        # 감정에 맞는 박스 색상 설정
        color = (0, 255, 0) if emotion == "Happy" else (0, 0, 255) if emotion in ["Angry", "Sad"] else (255, 255, 0)

        # 얼굴 영역에 감정 텍스트 표시
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return frame

# 웹캠 실행 로직
if st.session_state["webcam_active"]:
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        st.error("웹캠을 불러올 수 없습니다.")
        st.session_state["webcam_active"] = False
    else:
        stframe = st.image([])
        while st.session_state["webcam_active"]:
            ret, frame = video_capture.read()
            if not ret:
                st.error("웹캠 프레임을 가져올 수 없습니다.")
                st.session_state["webcam_active"] = False
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if not st.session_state["webcam_active"]:  # 예측 전에 다시 한 번 확인
                break

            frame = detect_emotion(frame)

            # 실시간 화면 업데이트
            stframe.image(frame, channels="RGB", use_container_width=True)

            time.sleep(0.1)  # CPU 부하 방지를 위해 약간의 딜레이 추가

        # 웹캠 종료 처리
        video_capture.release()
        st.success("웹캠이 정상적으로 종료되었습니다.")
