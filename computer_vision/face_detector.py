"""Real-time multi-face detection and 478-point landmark extraction via MediaPipe."""
from pathlib import Path
from typing import Any, Dict, List
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

class FaceDetector:
    def __init__(self, model_path: str = "models/face_landmarker.task", max_faces: int = 3):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"MediaPipe FaceLandmarker task file not found at '{path}'. "
                "Run `python models/download_models.py` first."
            )
        base_options = python.BaseOptions(model_asset_path=str(path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=max_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, bgr_frame: np.ndarray) -> List[Dict[str, Any]]:
        h, w, _ = bgr_frame.shape
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.landmarker.detect(mp_image)

        faces: List[Dict[str, Any]] = []
        if not detection_result.face_landmarks:
            return faces

        for idx, landmarks in enumerate(detection_result.face_landmarks):
            pts = np.array([(lm.x * w, lm.y * h, lm.z * w) for lm in landmarks], dtype=np.float32)
            x_min = max(0, int(np.min(pts[:, 0])) - 10)
            y_min = max(0, int(np.min(pts[:, 1])) - 10)
            x_max = min(w, int(np.max(pts[:, 0])) + 10)
            y_max = min(h, int(np.max(pts[:, 1])) + 10)

            blendshapes = {}
            if detection_result.face_blendshapes and idx < len(detection_result.face_blendshapes):
                blendshapes = {
                    b.category_name: round(float(b.score), 4)
                    for b in detection_result.face_blendshapes[idx]
                }

            faces.append(
                {
                    "face_id": idx,
                    "bbox": {"x": x_min, "y": y_min, "width": x_max - x_min, "height": y_max - y_min},
                    "landmarks_px": pts,
                    "landmarks_norm": [(float(lm.x), float(lm.y)) for lm in landmarks],
                    "blendshapes": blendshapes,
                }
            )
        return faces
