import sys
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import modules.holistic_module as hm
from tensorflow.keras.models import load_model
import time
from modules.utils import Vector_Normalization
from PIL import ImageFont, ImageDraw, Image

# 한글 폰트 설정
fontpath = "fonts/HMKMMAG.TTF"
font = ImageFont.truetype(fontpath, 40)

# 인식할 한글 자모 리스트
actions = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
           'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
           'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ']
seq_length = 10  # 시퀀스 길이 조정 (10 → 5)

# MediaPipe holistic model (손 검출 신뢰도 높이기)
detector = hm.HolisticDetector(min_detection_confidence=0.5)

# TensorFlow Lite 모델 로드
interpreter = tf.lite.Interpreter(model_path="models/multi_hand_gesture_classifier.tflite")
interpreter.allocate_tensors()

# 모델 입출력 정보 가져오기
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(0)  # 웹캠 열기

seq = []
action_seq = []
last_action = None

# 인식된 문자열 저장 변수
saved_string = ""
current_letter = None
current_letter_start_time = 0

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img = detector.findHolistic(img, draw=True)  # 전체 바디 검출
    _, right_hand_lmList = detector.findRighthandLandmark(img)  # 오른손 랜드마크 검출

    if right_hand_lmList is not None:
        joint = np.zeros((42, 2))
        for j, lm in enumerate(right_hand_lmList.landmark):
            joint[j] = [lm.x, lm.y]

        # 벡터 정규화 및 입력 데이터 변환
        vector, angle_label = Vector_Normalization(joint)
        d = np.concatenate([vector.flatten(), angle_label.flatten()])
        seq.append(d)

        if len(seq) < seq_length:
            continue

        # 시퀀스 입력 데이터 준비
        input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        y_pred = interpreter.get_tensor(output_details[0]['index'])
        i_pred = int(np.argmax(y_pred[0]))
        conf = y_pred[0][i_pred]

        # 신뢰도가 낮으면 무시
        if conf < 0.9:
            continue

        action = actions[i_pred]
        action_seq.append(action)

        if len(action_seq) < 3:
            continue

        # 안정적인 예측을 위해 최근 3프레임이 동일할 경우 확정
        this_action = '?'
        if action_seq[-1] == action_seq[-2] == action_seq[-3]:
            this_action = action
            if current_letter != this_action:
                current_letter = this_action
                current_letter_start_time = time.time()
            else:
                if time.time() - current_letter_start_time >= 0.8:  # 2초 → 0.7초로 단축
                    saved_string += this_action
                    current_letter = None

        # 화면에 인식된 문자열 및 현재 자모 표시
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text((10, 30), f'{saved_string}', font=font, fill=(255, 255, 255))  # 저장된 문자열
        if current_letter is not None:
            draw.text((10, 70), f'{current_letter}', font=font, fill=(0, 255, 0))  # 현재 인식 중인 자모
        img = np.array(img_pil)

    cv2.imshow('img', img)
    if cv2.waitKey(5) & 0xFF == 27:  # 1ms → 5ms로 변경
        break

cap.release()
cv2.destroyAllWindows()
