# streamlit run face_aging_streamlit.py

import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
import time

# Streamlit UI
st.title("🧑 AI 기반 얼굴 분석 웹")
st.write("웹캠을 사용하거나 이미지를 업로드하여 얼굴을 분석합니다.")

# 🔹 웹캠 & 업로드 이미지 저장 변수 초기화
if "webcam_active" not in st.session_state:
    st.session_state["webcam_active"] = False
if "captured_image" not in st.session_state:
    st.session_state["captured_image"] = None
if "uploaded_image" not in st.session_state:
    st.session_state["uploaded_image"] = None

# 🔹 웹캠 실행 & 종료 버튼 (비율 유지)
col1, col2 = st.columns([0.2, 0.8])

with col1:
    if st.button("📷 웹캠 실행", key="start_webcam"):
        st.session_state["webcam_active"] = True
        st.session_state["captured_image"] = None  # 기존 캡처 이미지 삭제
        st.session_state["uploaded_image"] = None  # 기존 업로드 이미지 삭제

with col2:
    if st.button("🛑 웹캠 종료", key="stop_webcam"):
        st.session_state["webcam_active"] = False

# 🔹 웹캠 실행 (실시간 스트리밍)
if st.session_state["webcam_active"]:
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        st.error("❌ 웹캠을 열 수 없습니다. 다른 프로그램에서 사용 중일 수 있습니다.")
        st.session_state["webcam_active"] = False
    else:
        stframe = st.empty()  # 🚀 실시간 프레임 업데이트 공간
        capture_button_placeholder = st.empty()  # 🚀 캡처 버튼 공간 유지

        # "📸 캡처" 버튼을 while 루프 밖에서 한 번만 생성
        capture_clicked = capture_button_placeholder.button("📸 캡처", key="capture_button")

        while st.session_state["webcam_active"]:
            ret, frame = video_capture.read()
            if not ret:
                st.error("❌ 웹캠 프레임을 읽을 수 없습니다.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame, channels="RGB", use_container_width=True)  # 실시간 스트리밍 유지

            # "📸 캡처" 버튼이 클릭되면 현재 화면 저장 후 종료
            if capture_clicked:
                st.session_state["captured_image"] = Image.fromarray(frame)  # 웹캠 이미지 저장
                st.session_state["uploaded_image"] = None  # 기존 업로드 이미지 삭제
                st.session_state["webcam_active"] = False  # 캡처 후 웹캠 종료
                st.rerun()  # UI 새로고침하여 웹캠 종료 및 이미지 표시

            time.sleep(0.03)  # 프레임 속도 조절 (CPU 부하 방지)

        video_capture.release()

# **업로드한 이미지 처리 (웹캠 실행 시 자동 삭제)**
uploaded_file = st.file_uploader("📂 이미지 업로드 (JPG, PNG)", type=["jpg", "png", "jpeg"])
if uploaded_file:
    st.session_state["uploaded_image"] = Image.open(uploaded_file)  # 업로드된 이미지 저장
    st.session_state["captured_image"] = None  # 기존 캡처 이미지 삭제

# 최종 분석할 얼굴 설정 (웹캠 캡처된 이미지가 우선)
image_to_analyze = st.session_state["captured_image"] or st.session_state["uploaded_image"]

# 분석할 얼굴 이미지 표시
if image_to_analyze:
    st.image(image_to_analyze, caption="📷 분석할 얼굴", width=250)

# 이미지 분석 함수 (성별 확률 변환 포함)
def analyze_face(image):
    image = np.array(image)
    try:
        analysis = DeepFace.analyze(image, actions=['age', 'gender', 'emotion', 'race'], enforce_detection=False)

        # 성별 확률 변환 (백분율 형태)
        gender_probs = analysis[0]['gender']
        male_prob = round(gender_probs['Man'], 2)
        female_prob = round(gender_probs['Woman'], 2)

        # 확률이 높은 값을 최종 성별로 표시
        final_gender = "남성" if male_prob > female_prob else "여성"

        return {
            "age": analysis[0]['age'],
            "gender": final_gender,
            "gender_prob": f"여성: {female_prob}% | 남성: {male_prob}%",
            "emotion": max(analysis[0]['emotion'], key=analysis[0]['emotion'].get),
            "race": max(analysis[0]['race'], key=analysis[0]['race'].get)
        }
    except Exception as e:
        st.error("얼굴을 감지할 수 없습니다. 다른 이미지를 사용해 주세요.")
        return None

# 분석 버튼
# if image_to_analyze:
#     if st.button("얼굴 분석하기", key="analyze_button"):
#         analysis = analyze_face(image_to_analyze)
#         if analysis:
#             st.success(f"🔢 **예상 나이:** {analysis['age']}세")
#             st.success(f"⚤ **성별:** {analysis['gender']} ({analysis['gender_prob']})")
#             st.success(f"🙂 **감정 상태:** {analysis['emotion']}")
#             st.success(f"🌎 **인종:** {analysis['race']}")

if image_to_analyze:
    if st.button("얼굴 분석하기", key="analyze_button"):
        analysis = analyze_face(image_to_analyze)
        if analysis:
            result_text = f"""
            <div style="padding: 10px; border-radius: 10px; background-color: #fcf0fb; border: 1px solid #f7c1f2;">
                <h4>🔎 분석 결과</h4>
                <p><b>🔢 예상 나이:</b> {analysis['age']}세</p>
                <p><b>⚤ 성별:</b> {analysis['gender']} ({analysis['gender_prob']})</p>
                <p><b>🙂 감정 상태:</b> {analysis['emotion']}</p>
                <p><b>🌎 인종:</b> {analysis['race']}</p>
            </div>
            """
            st.markdown(result_text, unsafe_allow_html=True)

