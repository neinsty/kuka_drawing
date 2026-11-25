import os
import time

import mediapipe as mp
import numpy as np
import cv2
# =========================================
# НАСТРОЙКИ
# =========================================

CAMERA_ID = 0

WIDTH = 640
HEIGHT = 480

MODEL_PATH = "face_landmarker.task"

# Толщина линий
LINE_THICKNESS = 1

# Контрастность контуров
CANNY_LOW = 65
CANNY_HIGH = 130

# Сглаживание
BLUR_SIZE = 5

# Стабилизация
SMOOTHING = 0.65

# =========================================
# MEDIAPIPE
# =========================================

BaseOptions = mp.tasks.BaseOptions
Vision = mp.tasks.vision

options = Vision.FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=Vision.RunningMode.VIDEO,
    num_faces=1
)

Connections = Vision.FaceLandmarksConnections

# Основные группы линий
FACE_GROUPS = [
    Connections.FACE_LANDMARKS_FACE_OVAL,
    Connections.FACE_LANDMARKS_LEFT_EYE,
    Connections.FACE_LANDMARKS_RIGHT_EYE,
    Connections.FACE_LANDMARKS_LEFT_EYEBROW,
    Connections.FACE_LANDMARKS_RIGHT_EYEBROW,
    Connections.FACE_LANDMARKS_LIPS,
    Connections.FACE_LANDMARKS_NOSE
]

# =========================================
# ПОСТРОЕНИЕ ЛИНИЙ ЛИЦА
# =========================================

def draw_face(canvas, landmarks, width, height):

    for group in FACE_GROUPS:

        for connection in group:

            p1 = landmarks[connection.start]
            p2 = landmarks[connection.end]

            x1 = int(p1.x * width)
            y1 = int(p1.y * height)

            x2 = int(p2.x * width)
            y2 = int(p2.y * height)

            cv2.line(
                canvas,
                (x1, y1),
                (x2, y2),
                0,
                LINE_THICKNESS,
                cv2.LINE_AA
            )

# =========================================
# ВЫДЕЛЕНИЕ ДОПОЛНИТЕЛЬНЫХ КОНТУРОВ
# =========================================

def draw_details(canvas, frame, landmarks):

    height, width = frame.shape[:2]

    # Границы лица
    points = np.array([
        [
            int(point.x * width),
            int(point.y * height)
        ]
        for point in landmarks
    ], dtype=np.int32)

    x_min = max(0, points[:, 0].min())
    y_min = max(0, points[:, 1].min())

    x_max = min(width, points[:, 0].max())
    y_max = min(height, points[:, 1].max())

    if x_max <= x_min or y_max <= y_min:
        return

    # Маска области лица
    mask = np.zeros((height, width), dtype=np.uint8)

    oval = np.array([
        [
            int(landmarks[i].x * width),
            int(landmarks[i].y * height)
        ]
        for i in [
            10, 338, 297, 332, 284, 251,
            389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377,
            152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127,
            162, 21, 54, 103, 67, 109
        ]
    ], dtype=np.int32)

    cv2.fillPoly(mask, [oval], 255)

    # Серая версия изображения
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Обработка текстуры
    blurred = cv2.GaussianBlur(
        gray,
        (BLUR_SIZE, BLUR_SIZE),
        0
    )

    # Дополнительные контуры
    edges = cv2.Canny(
        blurred,
        CANNY_LOW,
        CANNY_HIGH
    )

    # Оставляем только область лица
    edges = cv2.bitwise_and(
        edges,
        mask
    )

    # Удаляем мелкий шум
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

        length = cv2.arcLength(
            contour,
            False
        )

        if length < 25:
            continue

        simplified = cv2.approxPolyDP(
            contour,
            0.015 * length,
            False
        )

        cv2.drawContours(
            canvas,
            [simplified],
            -1,
            0,
            1,
            cv2.LINE_AA
        )

# =========================================
# ОСНОВНОЕ ПРЕОБРАЗОВАНИЕ
# =========================================

def make_portrait(frame, landmarker, timestamp):

    height, width = frame.shape[:2]

    # Создаём белый фон
    canvas = np.full(
        (height, width),
        255,
        dtype=np.uint8
    )

    # Подготовка изображения
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # Распознавание лица
    detection = landmarker.detect_for_video(
        mp_image,
        timestamp
    )

    if detection.face_landmarks:

        landmarks = detection.face_landmarks[0]

        # Геометрические линии
        draw_face(
            canvas,
            landmarks,
            width,
            height
        )

        # Художественные детали
        draw_details(
            canvas,
            frame,
            landmarks
        )

    return canvas

# =========================================
# ЗАПУСК КАМЕРЫ
# =========================================

camera = cv2.VideoCapture(CAMERA_ID)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    WIDTH
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    HEIGHT
)

if not camera.isOpened():
    raise RuntimeError("Камера не найдена")

os.makedirs("output", exist_ok=True)

previous_frame = None

# =========================================
# ОСНОВНОЙ ЦИКЛ
# =========================================

try:

    with Vision.FaceLandmarker.create_from_options(
        options
    ) as landmarker:

        while True:

            ret, frame = camera.read()

            if not ret:
                break

            frame = cv2.flip(frame, 1)

            frame = cv2.resize(
                frame,
                (WIDTH, HEIGHT)
            )

            timestamp = time.monotonic_ns() // 1_000_000

            result = make_portrait(
                frame,
                landmarker,
                timestamp
            )

            # Сглаживание между кадрами
            if previous_frame is not None:

                result = cv2.addWeighted(
                    result,
                    SMOOTHING,
                    previous_frame,
                    1 - SMOOTHING,
                    0
                )

            previous_frame = result.copy()

            # Вывод
            cv2.imshow(
                "Camera",
                frame
            )

            cv2.imshow(
                "Portrait Line Art",
                result
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):

                cv2.imwrite(
                    "output/portrait.png",
                    result
                )

                print("Portrait saved")

            if key == ord("q"):
                break

finally:

    camera.release()
    cv2.destroyAllWindows()