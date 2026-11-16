"""Eye landmark extraction, Eye Aspect Ratio (EAR), blink frequency, and iris gaze estimation."""
from collections import deque
from typing import Deque, Dict
import numpy as np

LEFT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380]
LEFT_IRIS_IDX = [468, 469, 470, 471, 472]
RIGHT_IRIS_IDX = [473, 474, 475, 476, 477]

class EyeTracker:
    def __init__(self, ear_threshold: float = 0.21, consec_frames: int = 2):
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
        self.closed_counter = 0
        self.total_blinks = 0
        self.blink_timestamps: Deque[float] = deque()

    @staticmethod
    def _compute_ear(eye_pts: np.ndarray) -> float:
        p1, p2, p3, p4, p5, p6 = eye_pts[:, :2]
        v1 = np.linalg.norm(p2 - p6)
        v2 = np.linalg.norm(p3 - p5)
        h = np.linalg.norm(p1 - p4)
        if h < 1e-6:
            return 0.0
        return float((v1 + v2) / (2.0 * h))

    def update(self, landmarks_px: np.ndarray, timestamp_ms: int) -> Dict:
        left_ear = self._compute_ear(landmarks_px[LEFT_EYE_EAR_IDX])
        right_ear = self._compute_ear(landmarks_px[RIGHT_EYE_EAR_IDX])
        avg_ear = (left_ear + right_ear) / 2.0

        is_blinking = avg_ear < self.ear_threshold
        if is_blinking:
            self.closed_counter += 1
        else:
            if self.closed_counter >= self.consec_frames:
                self.total_blinks += 1
                self.blink_timestamps.append(timestamp_ms / 1000.0)
            self.closed_counter = 0

        now_sec = timestamp_ms / 1000.0
        while self.blink_timestamps and (now_sec - self.blink_timestamps[0]) > 60.0:
            self.blink_timestamps.popleft()
        blinks_per_min = float(len(self.blink_timestamps))

        gaze_x, gaze_y = 0.5, 0.5
        if len(landmarks_px) >= 478:
            left_iris = np.mean(landmarks_px[LEFT_IRIS_IDX, :2], axis=0)
            right_iris = np.mean(landmarks_px[RIGHT_IRIS_IDX, :2], axis=0)

            l_outer, l_inner = landmarks_px[33, :2], landmarks_px[133, :2]
            l_top, l_bot = landmarks_px[159, :2], landmarks_px[145, :2]

            r_inner, r_outer = landmarks_px[362, :2], landmarks_px[263, :2]
            r_top, r_bot = landmarks_px[386, :2], landmarks_px[374, :2]

            lx = (left_iris[0] - l_outer[0]) / max(1e-5, (l_inner[0] - l_outer[0]))
            rx = (right_iris[0] - r_inner[0]) / max(1e-5, (r_outer[0] - r_inner[0]))
            gaze_x = float(np.clip((lx + rx) / 2.0, 0.0, 1.0))

            ly = (left_iris[1] - l_top[1]) / max(1e-5, (l_bot[1] - l_top[1]))
            ry = (right_iris[1] - r_top[1]) / max(1e-5, (r_bot[1] - r_top[1]))
            gaze_y = float(np.clip((ly + ry) / 2.0, 0.0, 1.0))

        direction = self._classify_gaze(gaze_x, gaze_y)
        approx_eye_contact = direction == "center" and not is_blinking

        return {
            "left_ear": round(left_ear, 4),
            "right_ear": round(right_ear, 4),
            "average_ear": round(avg_ear, 4),
            "is_blinking": is_blinking,
            "total_blinks": self.total_blinks,
            "blinks_per_minute": round(blinks_per_min, 1),
            "gaze_vector": {"x": round(gaze_x, 3), "y": round(gaze_y, 3)},
            "gaze_direction": direction,
            "approximate_eye_contact": approx_eye_contact,
            "left_iris_px": landmarks_px[468, :2].tolist() if len(landmarks_px) >= 478 else None,
            "right_iris_px": landmarks_px[473, :2].tolist() if len(landmarks_px) >= 478 else None,
        }

    @staticmethod
    def _classify_gaze(gx: float, gy: float) -> str:
        if gx < 0.38:
            return "right"
        if gx > 0.62:
            return "left"
        if gy < 0.35:
            return "up"
        if gy > 0.68:
            return "down"
        return "center"
