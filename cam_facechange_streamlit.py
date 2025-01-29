# streamlit run cam_facechange_streamlit.py


# 3. 윤곽선 없이 사각형도 없게 경계 부드럽게
import streamlit as st
import cv2
import random
import numpy as np
import mediapipe as mp
import time

# Mediapipe 초기화
mp_face_mesh = mp.solutions.face_mesh

# 얼굴 윤곽 마스크 생성 (경계 부드럽게 처리)
def create_face_mask(frame, landmarks, iw, ih):
    points = [
        (int(landmark.x * iw), int(landmark.y * ih))
        for landmark in landmarks
    ]
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)  # 초기화된 마스크 생성
    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)  # 다각형 윤곽선 채우기

    # 마스크 경계 부드럽게 처리
    mask = cv2.GaussianBlur(mask, (15, 15), 10)
    return mask

# 얼굴 교체 작업
def detect_and_swap_faces(frame):
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.7
    ) as face_mesh:
        # BGR -> RGB 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return frame  # 얼굴이 감지되지 않으면 원본 반환

        ih, iw, _ = frame.shape
        face_regions = []

        # 얼굴 영역 추출
        for face_landmarks in results.multi_face_landmarks:
            # 얼굴 전체 윤곽 (눈, 코, 턱 등 포함)
            landmarks = [face_landmarks.landmark[i] for i in range(468)]
            mask = create_face_mask(frame, landmarks, iw, ih)

            # 얼굴 경계 계산
            xs = [int(landmark.x * iw) for landmark in face_landmarks.landmark]
            ys = [int(landmark.y * ih) for landmark in face_landmarks.landmark]

            x_min, x_max = max(min(xs), 0), min(max(xs), iw - 1)
            y_min, y_max = max(min(ys), 0), min(max(ys), ih - 1)

            if (x_max - x_min) > 0 and (y_max - y_min) > 0:
                face_region = frame[y_min:y_max, x_min:x_max].copy()
                face_regions.append((face_region, mask, (x_min, y_min, x_max, y_max)))

        # 얼굴 교체 작업
        if len(face_regions) > 1:  # 두 명 이상일 때 랜덤으로 교체
            original_faces = [face for face, _, _ in face_regions]  # 원본 얼굴 리스트
            shuffled_faces = original_faces.copy()

            # 자기 자신 제외한 랜덤 섞기
            while True:
                random.shuffle(shuffled_faces)
                if all(original_faces[i] is not shuffled_faces[i] for i in range(len(original_faces))):
                    break

            # 랜덤 교체 수행
            for i, (face, mask, (x_min, y_min, x_max, y_max)) in enumerate(face_regions):
                h, w = y_max - y_min, x_max - x_min
                resized_face = cv2.resize(shuffled_faces[i], (w, h))  # 크기 조정

                # 얼굴 윤곽선 영역 교체
                target_roi = frame[y_min:y_max, x_min:x_max]

                # 기존 영역과 합성
                blended_face = cv2.addWeighted(target_roi, 0.5, resized_face, 0.5, 0)
                np.copyto(target_roi, blended_face, where=mask[y_min:y_max, x_min:x_max][:, :, None] > 0)

        return frame


# Streamlit
st.title("실시간 얼굴 바꾸기 (자연스러운 윤곽선 기반)")
st.text("웹캠을 켜고 두 사람 이상 들어오면 얼굴 윤곽선을 기반으로 랜덤으로 바꿉니다.")

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













# 2. 윤곽선 없이 좀 더 자연스럽게
# import streamlit as st
# import cv2
# import random
# import numpy as np
# import mediapipe as mp
# import time

# # Mediapipe 초기화
# mp_face_mesh = mp.solutions.face_mesh

# # 얼굴 윤곽 마스크 생성
# def create_face_mask(frame, landmarks, iw, ih):
#     points = [(int(landmark.x * iw), int(landmark.y * ih)) for landmark in landmarks]
#     mask = np.zeros(frame.shape[:2], dtype=np.uint8)
#     cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
#     return mask

# # 얼굴 감지 및 교체 함수
# def detect_and_swap_faces(frame):
#     with mp_face_mesh.FaceMesh(
#         static_image_mode=False,
#         max_num_faces=5,
#         refine_landmarks=True,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.7  # 추적 신뢰도 증가
#     ) as face_mesh:
#         # BGR -> RGB 변환
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = face_mesh.process(rgb_frame)

#         if not results.multi_face_landmarks:
#             return frame  # 얼굴이 감지되지 않으면 원본 반환

#         ih, iw, _ = frame.shape
#         face_regions = []

#         # 얼굴 영역 추출 및 박스 표시
#         for face_landmarks in results.multi_face_landmarks:
#             # 얼굴 윤곽선(왼쪽과 오른쪽 눈썹, 턱)을 기준으로 마스크 생성
#             landmarks = [face_landmarks.landmark[i] for i in list(range(468))]
#             mask = create_face_mask(frame, landmarks, iw, ih)

#             xs = [int(landmark.x * iw) for landmark in face_landmarks.landmark]
#             ys = [int(landmark.y * ih) for landmark in face_landmarks.landmark]

#             # 얼굴 경계 계산
#             x_min, x_max = max(min(xs), 0), min(max(xs), iw - 1)
#             y_min, y_max = max(min(ys), 0), min(max(ys), ih - 1)

#             if (x_max - x_min) > 0 and (y_max - y_min) > 0:
#                 face_region = frame[y_min:y_max, x_min:x_max].copy()
#                 face_regions.append((face_region, mask, (x_min, y_min, x_max, y_max)))

#         # 얼굴 교체 작업
#         if len(face_regions) > 1:  # 두 명 이상일 때 랜덤으로 교체
#             original_faces = [face for face, _, _ in face_regions]  # 원본 얼굴 리스트
#             shuffled_faces = original_faces.copy()

#             # 자기 자신 제외한 랜덤 섞기
#             while True:
#                 random.shuffle(shuffled_faces)
#                 if all(original_faces[i] is not shuffled_faces[i] for i in range(len(original_faces))):
#                     break

#             # 랜덤 교체 수행
#             for i, (face, mask, (x_min, y_min, x_max, y_max)) in enumerate(face_regions):
#                 h, w = y_max - y_min, x_max - x_min
#                 resized_face = cv2.resize(shuffled_faces[i], (w, h))  # 크기 조정

#                 # 얼굴 밝기 및 색상 조정
#                 alpha = 0.7  # 블렌딩 강도
#                 blended_face = cv2.addWeighted(frame[y_min:y_max, x_min:x_max], 1 - alpha, resized_face, alpha, 0)

#                 # 얼굴 합성
#                 frame[y_min:y_max, x_min:x_max] = blended_face

#         return frame

# # Streamlit
# st.title("실시간 얼굴 바꾸기 (자연스러운 랜덤 교체)")
# st.text("웹캠을 켜고 두 사람 이상 들어오면 얼굴을 자연스럽게 랜덤으로 바꿉니다.")

# # 상태 변수 초기화
# if "streaming" not in st.session_state:
#     st.session_state.streaming = False

# # 버튼 UI
# col1, _, col2 = st.columns([0.4, 0.07, 2])  # 가운데 빈 열로 간격 조절
# with col1:
#     start_button = st.button("웹캠 시작")
# with col2:
#     stop_button = st.button("웹캠 끄기")

# # 웹캠 스트리밍 제어
# if start_button:
#     st.session_state.streaming = True

# if stop_button:
#     st.session_state.streaming = False

# # 스트리밍 실행
# if st.session_state.streaming:
#     cap = cv2.VideoCapture(0)

#     try:
#         if not cap.isOpened():
#             st.error("웹캠을 찾을 수 없습니다. 웹캠이 연결되었는지 확인하세요.")
#             st.session_state.streaming = False
#         else:
#             frame_placeholder = st.empty()
#             while st.session_state.streaming:
#                 ret, frame = cap.read()
#                 if not ret:
#                     st.error("프레임을 읽을 수 없습니다. 스트리밍을 다시 시작하세요.")
#                     st.session_state.streaming = False
#                     break

#                 # 얼굴 감지 및 교체
#                 swapped_frame = detect_and_swap_faces(frame)

#                 # 결과 프레임 표시
#                 frame_placeholder.image(cv2.cvtColor(swapped_frame, cv2.COLOR_BGR2RGB), channels="RGB")
#                 time.sleep(0.03)  # UI 반응 속도를 위한 딜레이
#     finally:
#         cap.release()












# 1. 윤곽선 있는 랜덤 교체
# import streamlit as st
# import cv2
# import random
# import numpy as np
# import mediapipe as mp
# import time

# # Mediapipe 초기화
# mp_face_mesh = mp.solutions.face_mesh

# # 얼굴 감지 및 교체 함수
# def detect_and_swap_faces(frame):
#     with mp_face_mesh.FaceMesh(
#         static_image_mode=False,
#         max_num_faces=5,
#         refine_landmarks=True,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5
#     ) as face_mesh:
#         # BGR -> RGB 변환
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = face_mesh.process(rgb_frame)

#         if not results.multi_face_landmarks:
#             return frame  # 얼굴이 감지되지 않으면 원본 반환

#         ih, iw, _ = frame.shape
#         face_regions = []

#         # 얼굴 영역 추출 및 박스 표시
#         for face_landmarks in results.multi_face_landmarks:
#             xs = [int(landmark.x * iw) for landmark in face_landmarks.landmark]
#             ys = [int(landmark.y * ih) for landmark in face_landmarks.landmark]

#             # 얼굴 경계 계산
#             x_min, x_max = max(min(xs), 0), min(max(xs), iw - 1)
#             y_min, y_max = max(min(ys), 0), min(max(ys), ih - 1)

#             if (x_max - x_min) > 0 and (y_max - y_min) > 0:
#                 face_region = frame[y_min:y_max, x_min:x_max].copy()
#                 face_regions.append((face_region, (x_min, y_min, x_max, y_max)))

#                 # 얼굴 박스 그리기
#                 cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

#         # 얼굴 교체 작업
#         if len(face_regions) == 2:  # 두 명일 때 강제 1:1 교체
#             face1, coords1 = face_regions[0]
#             face2, coords2 = face_regions[1]

#             # 얼굴 1 -> 얼굴 2 위치
#             x1_min, y1_min, x1_max, y1_max = coords1
#             x2_min, y2_min, x2_max, y2_max = coords2

#             resized_face1 = cv2.resize(face1, (x2_max - x2_min, y2_max - y2_min))
#             resized_face2 = cv2.resize(face2, (x1_max - x1_min, y1_max - y1_min))

#             frame[y1_min:y1_max, x1_min:x1_max] = resized_face2
#             frame[y2_min:y2_max, x2_min:x2_max] = resized_face1

#         elif len(face_regions) > 2:  # 세 명 이상일 때 랜덤으로 섞기
#             original_faces = [face for face, _ in face_regions]  # 원본 얼굴 이미지 리스트
#             shuffled_faces = original_faces.copy()

#             # 자기 자신 제외한 랜덤 섞기
#             while True:
#                 random.shuffle(shuffled_faces)
#                 if all(original_faces[i] is not shuffled_faces[i] for i in range(len(original_faces))):
#                     break

#             # 랜덤 교체 수행
#             for i, (face, (x_min, y_min, x_max, y_max)) in enumerate(face_regions):
#                 h, w = y_max - y_min, x_max - x_min
#                 resized_face = cv2.resize(shuffled_faces[i], (w, h))  # 크기 조정

#                 # 얼굴 합성
#                 frame[y_min:y_max, x_min:x_max] = resized_face

#         return frame

# # Streamlit
# st.title("실시간 얼굴 바꾸기 (랜덤 교체)")
# st.text("웹캠을 켜고 두 사람 이상 들어오면 얼굴을 랜덤으로 바꿉니다.")

# # 상태 변수 초기화
# if "streaming" not in st.session_state:
#     st.session_state.streaming = False

# # 버튼 UI
# col1, _, col2 = st.columns([0.4, 0.07, 2])  # 가운데 빈 열로 간격 조절
# with col1:
#     start_button = st.button("웹캠 시작")
# with col2:
#     stop_button = st.button("웹캠 끄기")

# # 웹캠 스트리밍 제어
# if start_button:
#     st.session_state.streaming = True

# if stop_button:
#     st.session_state.streaming = False

# # 스트리밍 실행
# if st.session_state.streaming:
#     cap = cv2.VideoCapture(0)

#     try:
#         if not cap.isOpened():
#             st.error("웹캠을 찾을 수 없습니다. 웹캠이 연결되었는지 확인하세요.")
#             st.session_state.streaming = False
#         else:
#             frame_placeholder = st.empty()
#             while st.session_state.streaming:
#                 ret, frame = cap.read()
#                 if not ret:
#                     st.error("프레임을 읽을 수 없습니다. 스트리밍을 다시 시작하세요.")
#                     st.session_state.streaming = False
#                     break

#                 # 얼굴 감지 및 교체
#                 swapped_frame = detect_and_swap_faces(frame)

#                 # 결과 프레임 표시
#                 frame_placeholder.image(cv2.cvtColor(swapped_frame, cv2.COLOR_BGR2RGB), channels="RGB")
#                 time.sleep(0.03)  # UI 반응 속도를 위한 딜레이
#     finally:
#         cap.release()