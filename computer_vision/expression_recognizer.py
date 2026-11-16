"""
Visual facial expression classification.
Supports ViT image classification on cropped face ROI with blendshape fallback.
Outputs visual expression probabilities — not psychological or emotional ground truth.
"""
from typing import Dict
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

SUPPORTED_EXPRESSIONS = ["Happy", "Sad", "Angry", "Surprised", "Fearful", "Disgusted", "Neutral"]

LABEL_MAP = {
    "happy": "Happy",
    "sad": "Sad",
    "angry": "Angry",
    "surprise": "Surprised",
    "surprised": "Surprised",
    "fear": "Fearful",
    "fearful": "Fearful",
    "disgust": "Disgusted",
    "disgusted": "Disgusted",
    "neutral": "Neutral",
}

class ExpressionRecognizer:
    def __init__(self, model_name: str = "trpakov/vit-face-expression", use_pretrained: bool = True):
        self.use_pretrained = use_pretrained
        self.processor = None
        self.model = None
        if use_pretrained:
            try:
                self.processor = AutoImageProcessor.from_pretrained(model_name)
                self.model = AutoModelForImageClassification.from_pretrained(model_name)
                self.model.eval()
            except Exception:
                self.use_pretrained = False

    def predict(self, bgr_frame: np.ndarray, bbox: Dict[str, int], blendshapes: Dict[str, float]) -> Dict:
        if self.use_pretrained and self.model is not None and self.processor is not None:
            x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
            crop = bgr_frame[max(0, y): y + max(1, h), max(0, x): x + max(1, w)]
            if crop.size > 0:
                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_crop)
                inputs = self.processor(images=pil_img, return_tensors="pt")
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()

                scores = {label: 0.0 for label in SUPPORTED_EXPRESSIONS}
                for idx, prob in enumerate(probs):
                    raw_label = self.model.config.id2label[idx].lower()
                    mapped = LABEL_MAP.get(raw_label)
                    if mapped:
                        scores[mapped] = round(float(prob), 4)

                dominant = max(scores, key=scores.get)
                return {
                    "dominant_expression": dominant,
                    "confidence": scores[dominant],
                    "probabilities": scores,
                    "inference_mode": "vit-face-expression",
                }

        return self._predict_from_blendshapes(blendshapes)

    def _predict_from_blendshapes(self, bs: Dict[str, float]) -> Dict:
        smile = (bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0)) / 2.0
        frown = (bs.get("mouthFrownLeft", 0.0) + bs.get("mouthFrownRight", 0.0)) / 2.0
        brow_down = (bs.get("browDownLeft", 0.0) + bs.get("browDownRight", 0.0)) / 2.0
        brow_up = bs.get("browInnerUp", 0.0)
        jaw_open = bs.get("jawOpen", 0.0)
        nose_sneer = (bs.get("noseSneerLeft", 0.0) + bs.get("noseSneerRight", 0.0)) / 2.0
        eye_wide = (bs.get("eyeWideLeft", 0.0) + bs.get("eyeWideRight", 0.0)) / 2.0

        raw = {
            "Happy": smile * 1.4,
            "Sad": (frown * 0.8 + brow_up * 0.4),
            "Angry": brow_down * 1.3,
            "Surprised": (jaw_open * 0.7 + eye_wide * 0.7),
            "Fearful": (eye_wide * 0.6 + brow_up * 0.6),
            "Disgusted": nose_sneer * 1.4,
            "Neutral": 0.35,
        }
        total = sum(raw.values()) or 1.0
        probs = {k: round(v / total, 4) for k, v in raw.items()}
        dominant = max(probs, key=probs.get)
        return {
            "dominant_expression": dominant,
            "confidence": probs[dominant],
            "probabilities": probs,
            "inference_mode": "mediapipe-blendshapes",
        }
