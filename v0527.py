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
from datetime import datetime

# 자모 리스트
CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ',
                'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
             'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
JONG_LIST = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ',
             'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
             'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

def assemble_hangul(chosung, jung, jong=''):
    try:
        cho_index = CHOSUNG_LIST.index(chosung)
        jung_index = JUNG_LIST.index(jung)
    except ValueError:
        return chosung + jung + jong
    jong_index = JONG_LIST.index(jong) if jong in JONG_LIST else 0
    code = 0xAC00 + (cho_index * 21 * 28) + (jung_index * 28) + jong_index
    return chr(code)

def combine_jamos_custom(zamo_list):
    result = ''
    i = 0
    while i < len(zamo_list):
        if i + 1 < len(zamo_list):
            cho = zamo_list[i]
            jung = zamo_list[i + 1]
            jong = ''
            # 종성 조건 확인
            if (
                i + 2 < len(zamo_list) and 
                zamo_list[i + 2] in JONG_LIST[1:] and
                (i + 3 >= len(zamo_list) or zamo_list[i + 3] in CHOSUNG_LIST)
            ):
                jong = zamo_list[i + 2]
                i += 3
            else:
                i += 2
            result += assemble_hangul(cho, jung, jong)
        else:
            result += zamo_list[i]
            i += 1
    return result

# 폰트
fontpath = "fonts/HMKMMAG.TTF"
font = ImageFont.truetype(fontpath, 40)

actions = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
           'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
           'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ']
seq_length = 10

detector = hm.HolisticDetector(min_detection_confidence=0.5)

interpreter = tf.lite.Interpreter(model_path="models/multi_hand_gesture_classifier.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(0)

seq = []
action_seq = []
last_action = None

zamo_list = []
current_letter = None
current_letter_start_time = 0
recognized_history = {}  # 자모별 마지막 인식 시간

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img = detector.findHolistic(img, draw=True)
    _, right_hand_lmList = detector.findRighthandLandmark(img)

    if right_hand_lmList is not None:
        joint = np.zeros((42, 2))
        for j, lm in enumerate(right_hand_lmList.landmark):
            joint[j] = [lm.x, lm.y]

        vector, angle_label = Vector_Normalization(joint)
        d = np.concatenate([vector.flatten(), angle_label.flatten()])
        seq.append(d)

        if len(seq) < seq_length:
            continue

        input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        y_pred = interpreter.get_tensor(output_details[0]['index'])
        i_pred = int(np.argmax(y_pred[0]))
        conf = y_pred[0][i_pred]

        if conf < 0.9:
            continue

        action = actions[i_pred]
        action_seq.append(action)

        if len(action_seq) < 3:
            continue

        this_action = '?'
        if action_seq[-1] == action_seq[-2] == action_seq[-3]:
            this_action = action

            now = time.time()
            last_time = recognized_history.get(this_action, 0)

            if current_letter != this_action:
                current_letter = this_action
                current_letter_start_time = now
            else:
                if now - current_letter_start_time >= 0.8 and now - last_time >= 2.0:
                    zamo_list.append(this_action)
                    recognized_history[this_action] = now
                    current_letter = None

    # 화면 출력
    assembled_text = combine_jamos_custom(zamo_list)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((10, 30), f'{assembled_text}', font=font, fill=(255, 255, 255))
    if current_letter is not None:
        draw.text((10, 70), f'{current_letter}', font=font, fill=(0, 255, 0))
    img = np.array(img_pil)

    cv2.imshow('img', img)
    key = cv2.waitKey(5) & 0xFF

    if key == 27:
        break

    if key == ord('s'):
        if assembled_text:
            filename = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(assembled_text)
            print(f"[✅ 저장됨] {filename}")

cap.release()
cv2.destroyAllWindows()
