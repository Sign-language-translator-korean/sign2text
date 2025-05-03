import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import modules.holistic_module as hm
from modules.utils import Vector_Normalization

# 모델 로드
interpreter = tf.lite.Interpreter(model_path="models/multi_hand_gesture_classifier.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# MediaPipe 세팅
detector = hm.HolisticDetector(min_detection_confidence=0.5)

# 인식할 자모 리스트
actions = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
           'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
           'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ']

seq_length = 10  # 시퀀스 길이

def process_video(video_path, output_path="output.mp4"):
    cap = cv2.VideoCapture(video_path)

    # 비디오 정보
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 결과 비디오 저장 세팅
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    seq = []
    action_seq = []
    last_action = None
    saved_string = ""
    current_letter = None
    current_letter_frame = 0
    frame_count = 0

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        frame_count += 1

        img = detector.findHolistic(img, draw=False)
        _, right_hand_lmList = detector.findRighthandLandmark(img)

        display_text = saved_string

        if right_hand_lmList is not None:
            joint = np.zeros((42, 2))
            for j, lm in enumerate(right_hand_lmList.landmark):
                joint[j] = [lm.x, lm.y]

            vector, angle_label = Vector_Normalization(joint)
            d = np.concatenate([vector.flatten(), angle_label.flatten()])
            seq.append(d)

            if len(seq) >= seq_length:
                input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                y_pred = interpreter.get_tensor(output_details[0]['index'])
                i_pred = int(np.argmax(y_pred[0]))
                conf = y_pred[0][i_pred]

                if conf > 0.9:
                    action = actions[i_pred]
                    action_seq.append(action)

                    if len(action_seq) >= 3 and action_seq[-1] == action_seq[-2] == action_seq[-3]:
                        this_action = action
                        if current_letter != this_action:
                            current_letter = this_action
                            current_letter_frame = frame_count
                        else:
                            if frame_count - current_letter_frame >= 10:
                                saved_string += this_action
                                current_letter = None

                        display_text = saved_string + f" ({this_action})"

        # 자막 출력
        cv2.putText(img, display_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

        out.write(img)
        # 원한다면 실시간 디스플레이도 가능
        # cv2.imshow('Video', img)
        # if cv2.waitKey(1) == ord('q'):
        #     break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return saved_string

# 사용 예시
if __name__ == "__main__":
    video_file = "C:/Users/KHJ/Desktop/idea/test/Sign_Language_Translation/videoplayback.mp4"
    result_text = process_video(video_file, output_path="output_annotated.mp4")
    print("최종 인식 결과:", result_text)
