# ABOUTME: Face detection and embedding extraction using InsightFace (RetinaFace + ArcFace)
# ABOUTME: Supports detection-only mode (for SORT) and detection+recognition mode (for DeepSORT/StrongSORT)

from __future__ import annotations

import os

import numpy as np
from insightface.app import FaceAnalysis

from .types import FaceDetection


class FaceDetector:
    """Wraps InsightFace FaceAnalysis for face detection with optional embedding extraction."""

    def __init__(
        self,
        extract_embeddings: bool = False,
        det_size: tuple[int, int] = (640, 640),
        conf_threshold: float = 0.5,
        device: str = "cuda",
        model_dir: str = "models",
    ):
        os.environ["INSIGHTFACE_HOME"] = model_dir

        allowed = ["detection"]
        if extract_embeddings:
            allowed.append("recognition")

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )

        self.app = FaceAnalysis(
            name="buffalo_l",
            root=model_dir,
            allowed_modules=allowed,
            providers=providers,
        )
        ctx_id = 0 if device == "cuda" else -1
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        self.conf_threshold = conf_threshold
        self.extract_embeddings = extract_embeddings

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        """Detect faces in a frame, optionally extracting embeddings."""
        faces = self.app.get(frame)
        results = []
        for face in faces:
            if face.det_score < self.conf_threshold:
                continue
            embedding = None
            if self.extract_embeddings and hasattr(face, "embedding") and face.embedding is not None:
                embedding = face.embedding.astype(np.float32)
            results.append(
                FaceDetection(
                    bbox_xyxy=face.bbox.astype(np.float32),
                    confidence=float(face.det_score),
                    embedding=embedding,
                )
            )
        return results
