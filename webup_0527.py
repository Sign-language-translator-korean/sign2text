import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import modules.holistic_module as hm
from modules.utils import Vector_Normalization
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image
from datetime import datetime
import time

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

# 설정
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

def process_video(video_path, output_path="output_result.mp4"):
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    seq = []
    action_seq = []
    last_action = None
    zamo_list = []
    current_letter = None
    current_letter_start_time = 0
    recognized_history = {}

    frame_time = 1 / fps if fps > 0 else 1 / 30
    now_timestamp = 0

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        img = detector.findHolistic(img, draw=False)
        _, right_hand_lmList = detector.findRighthandLandmark(img)

        if right_hand_lmList is not None:
            joint = np.zeros((42, 2))
            for j, lm in enumerate(right_hand_lmList.landmark):
                joint[j] = [lm.x, lm.y]

            vector, angle_label = Vector_Normalization(joint)
            d = np.concatenate([vector.flatten(), angle_label.flatten()])
            seq.append(d)

            if len(seq) < seq_length:
                out.write(img)
                now_timestamp += frame_time
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            y_pred = interpreter.get_tensor(output_details[0]['index'])
            i_pred = int(np.argmax(y_pred[0]))
            conf = y_pred[0][i_pred]

            if conf < 0.9:
                out.write(img)
                now_timestamp += frame_time
                continue

            action = actions[i_pred]
            action_seq.append(action)

            if len(action_seq) >= 3 and action_seq[-1] == action_seq[-2] == action_seq[-3]:
                this_action = action
                last_time = recognized_history.get(this_action, -999)
                if current_letter != this_action:
                    current_letter = this_action
                    current_letter_start_time = now_timestamp
                else:
                    if now_timestamp - current_letter_start_time >= 0.8 and now_timestamp - last_time >= 2.0:
                        zamo_list.append(this_action)
                        recognized_history[this_action] = now_timestamp
                        current_letter = None

        # 화면 출력
        assembled_text = combine_jamos_custom(zamo_list)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text((10, 30), f'{assembled_text}', font=font, fill=(255, 255, 255))
        if current_letter is not None:
            draw.text((10, 70), f'{current_letter}', font=font, fill=(0, 255, 0))
        img = np.array(img_pil)

        out.write(img)
        now_timestamp += frame_time

    cap.release()
    out.release()
    print(f"[✅ 완료] 자막 영상 저장됨: {output_path}")
    return combine_jamos_custom(zamo_list)

# 실행 예시
if __name__ == "__main__":
    video_file = "C:/Users/KHJ/Desktop/idea/test/Sign_Language_Translation/videoplayback.mp4"
    final_text = process_video(video_file, output_path="output_annotated.mp4")
    print("최종 인식 결과:", final_text)
