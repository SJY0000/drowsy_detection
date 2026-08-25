import cv2
import numpy as np
from openvino.runtime import Core

# ===== 설정 =====
DEVICE = "GPU.1"        # Arc A770. 문제 생기면 "CPU"로 바꿔서 테스트
CONF_THRESHOLD = 0.5    # 이 신뢰도 이상인 얼굴만 사용
CAMERA_ID = 0           # 기본 웹캠
FONT = cv2.FONT_HERSHEY_SIMPLEX

FACE_MODEL = "models/intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml"
HEAD_MODEL = "models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml"
LMK_MODEL  = "models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml"
EYE_MODEL  = "models/public/open-closed-eye-0001/FP32/open-closed-eye-0001.xml"

# ===== 모델 로드 =====
core = Core()

def load_model(path):
    return core.compile_model(core.read_model(path), DEVICE)

face_net = load_model(FACE_MODEL)
head_net = load_model(HEAD_MODEL)
lmk_net  = load_model(LMK_MODEL)
eye_net  = load_model(EYE_MODEL)

def input_size(net):
    _, _, h, w = net.input(0).shape
    return w, h

FACE_W, FACE_H = input_size(face_net)
HEAD_W, HEAD_H = input_size(head_net)
LMK_W,  LMK_H  = input_size(lmk_net)
EYE_W,  EYE_H  = input_size(eye_net)

def preprocess(image, w, h):
    resized = cv2.resize(image, (w, h))
    blob = resized.transpose(2, 0, 1)      # HWC -> CHW
    blob = np.expand_dims(blob, 0)         # -> (1, 3, H, W)
    return blob


# =====================================================================
#  detect(frame) 반환 형태 (이 딕셔너리를 졸음 판단 코드로 넘김)
# ---------------------------------------------------------------------
#  ● 운전자(얼굴) 검출된 경우:
#    {
#        "detected": True,
#        "box": (xmin, ymin, xmax, ymax),   # 얼굴 위치 (픽셀 좌표)
#        "confidence": 0.98,                # 얼굴 검출 신뢰도 (0~1)
#        "head_pose": {
#            "yaw":   3.2,    # 좌우 회전 (도)
#            "pitch": -12.5,  # 상하(고개 숙임) (도) ← 고개 숙이면 값이 변함
#            "roll":  1.1,    # 좌우 기울기 (도)
#        },
#        "left_eye":  {"state": "Open",   "closed_prob": 0.04},
#        "right_eye": {"state": "Closed", "closed_prob": 0.91},
#        #  state: "Open" / "Closed"
#        #  closed_prob: 감은 확률 0~1 (1에 가까울수록 감음)
#    }
#
#  ● 운전자(얼굴) 안 잡힌 경우:
#    {
#        "detected": False,
#        "box": None, "confidence": 0.0,
#        "head_pose": None, "left_eye": None, "right_eye": None,
#    }
#
#  * 참고: 눈이 화면 가장자리에 걸리면 left_eye / right_eye 가 None 일 수 있음.
#    판단 코드에서 None 여부를 먼저 확인하는 게 안전함.
# =====================================================================

def empty_result():
    return {
        "detected": False,
        "box": None,
        "confidence": 0.0,
        "head_pose": None,
        "left_eye": None,
        "right_eye": None,
    }

def analyze_eye(frame, cx, cy, half, fw, fh):
    # 눈 중심(cx, cy) 주변을 잘라 눈 상태를 판별
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(fw, cx + half), min(fh, cy + half)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    blob = preprocess(crop, EYE_W, EYE_H)
    out = eye_net([blob])[eye_net.output(0)].reshape(-1)  # [open, closed]
    closed_prob = float(out[1])                            # index 1 = 감음
    state = "Open" if out[0] > out[1] else "Closed"
    return {"state": state, "closed_prob": round(closed_prob, 3)}

def detect(frame):
    """한 프레임을 받아 운전자(가장 큰 얼굴) 감지 결과 딕셔너리를 반환."""
    fh, fw = frame.shape[:2]

    # --- 얼굴 검출 ---
    face_blob = preprocess(frame, FACE_W, FACE_H)
    faces = face_net([face_blob])[face_net.output(0)][0][0]
    # 각 검출: [image_id, label, conf, x_min, y_min, x_max, y_max] (좌표 0~1)

    # --- 신뢰도 넘는 얼굴 중 '가장 큰 얼굴(운전자)' 하나만 선택 ---
    best = None
    best_area = 0
    for det in faces:
        conf = float(det[2])
        if conf < CONF_THRESHOLD:
            continue
        xmin = max(0, int(det[3] * fw))
        ymin = max(0, int(det[4] * fh))
        xmax = min(fw, int(det[5] * fw))
        ymax = min(fh, int(det[6] * fh))
        area = (xmax - xmin) * (ymax - ymin)
        if area > best_area:
            best_area = area
            best = (conf, xmin, ymin, xmax, ymax)

    if best is None:
        return empty_result()   # 얼굴 없음

    conf, xmin, ymin, xmax, ymax = best
    face_crop = frame[ymin:ymax, xmin:xmax]
    if face_crop.size == 0:
        return empty_result()
    face_cw = xmax - xmin
    face_ch = ymax - ymin

    # --- 머리 각도 ---
    head_blob = preprocess(face_crop, HEAD_W, HEAD_H)
    hr = head_net([head_blob])
    head_pose = {
        "yaw":   round(float(hr[head_net.output("angle_y_fc")][0][0]), 1),
        "pitch": round(float(hr[head_net.output("angle_p_fc")][0][0]), 1),
        "roll":  round(float(hr[head_net.output("angle_r_fc")][0][0]), 1),
    }

    # --- 랜드마크로 눈 위치 찾기 ---
    lmk_blob = preprocess(face_crop, LMK_W, LMK_H)
    lmk = lmk_net([lmk_blob])[lmk_net.output(0)].reshape(-1)
    # lmk[0],[1] = 왼눈(x,y) / lmk[2],[3] = 오른눈(x,y)  (얼굴 crop 기준 0~1)
    left_x  = xmin + int(lmk[0] * face_cw)
    left_y  = ymin + int(lmk[1] * face_ch)
    right_x = xmin + int(lmk[2] * face_cw)
    right_y = ymin + int(lmk[3] * face_ch)

    # --- 눈 상태 ---
    eye_half = max(8, int(face_cw * 0.15))
    left_eye  = analyze_eye(frame, left_x,  left_y,  eye_half, fw, fh)
    right_eye = analyze_eye(frame, right_x, right_y, eye_half, fw, fh)

    return {
        "detected": True,
        "box": (xmin, ymin, xmax, ymax),
        "confidence": round(conf, 3),
        "head_pose": head_pose,
        "left_eye": left_eye,
        "right_eye": right_eye,
    }


def draw_result(frame, result):
    """감지 결과를 화면에 표시 (확인용). 판단 로직과는 별개."""
    if not result["detected"]:
        cv2.putText(frame, "No face", (10, 30), FONT, 0.7, (0, 0, 255), 2)
        return
    xmin, ymin, xmax, ymax = result["box"]
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    hp = result["head_pose"]
    cv2.putText(frame, f"Y:{hp['yaw']:.0f} P:{hp['pitch']:.0f} R:{hp['roll']:.0f}",
                (xmin, ymax + 20), FONT, 0.55, (255, 255, 0), 2)

    le, re = result["left_eye"], result["right_eye"]
    le_txt = le["state"] if le else "-"
    re_txt = re["state"] if re else "-"
    cv2.putText(frame, f"L:{le_txt}  R:{re_txt}",
                (xmin, ymin - 10), FONT, 0.55, (0, 255, 255), 2)


# ===== 메인 루프 =====
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다. 카메라 연결을 확인하세요.")
    exit()

print("EYEON 감지 시작 (종료하려면 q)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽지 못했습니다.")
        break

    # 매 프레임 감지 결과(딕셔너리) 계산
    result = detect(frame)

    # =================================================================
    #  여기서 result 를 졸음 판단 코드로 넘기면 됨.
    #  예) drowsiness_check(result)
    #  result["detected"] 로 얼굴 유무 확인 후,
    #  result["left_eye"]["closed_prob"], result["head_pose"]["pitch"] 등 사용
    #
    #  예시) 사용법
    #  result = detect(frame)
    #  if result["detected"]:
    #      pitch = result["head_pose"]["pitch"]
    #      left_closed = result["left_eye"]["closed_prob"]
    #      # 여기서 졸음 판단...
    # =================================================================

    # 화면 표시 (확인용)
    draw_result(frame, result)
    cv2.imshow("EYEON - Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
