"""3D Head Pose Estimation (Yaw, Pitch, Roll) using OpenCV solvePnP."""
from typing import Dict, Tuple
import cv2
import numpy as np

MODEL_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)

LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]

class HeadPoseEstimator:
    def __init__(self, yaw_threshold: float = 22.0, pitch_threshold: float = 18.0):
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold

    def estimate(self, landmarks_px: np.ndarray, frame_shape: Tuple[int, int, int]) -> Dict:
        h, w, _ = frame_shape
        image_points = np.array([landmarks_px[i, :2] for i in LANDMARK_INDICES], dtype=np.float64)

        focal_length = float(w)
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vec, translation_vec = cv2.solvePnP(
            MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "facing_camera": True, "orientation": "center"}

        rot_mat, _ = cv2.Rodrigues(rotation_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)

        pitch = float(angles[0])
        yaw = float(angles[1])
        roll = float(angles[2])

        facing_camera = abs(yaw) <= self.yaw_threshold and abs(pitch) <= self.pitch_threshold
        orientation = "center"
        if yaw > self.yaw_threshold:
            orientation = "turned_right"
        elif yaw < -self.yaw_threshold:
            orientation = "turned_left"
        elif pitch > self.pitch_threshold:
            orientation = "tilted_up"
        elif pitch < -self.pitch_threshold:
            orientation = "tilted_down"

        return {
            "yaw": round(yaw, 2),
            "pitch": round(pitch, 2),
            "roll": round(roll, 2),
            "facing_camera": facing_camera,
            "orientation": orientation,
        }
