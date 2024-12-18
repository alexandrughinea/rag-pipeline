from typing import List

import cv2
import numpy as np
from pytesseract import pytesseract


class VideoProcessor:
    def __init__(self, sample_rate: int = 1, similarity_threshold: float = 0.9):
        self.sample_rate = sample_rate
        self.similarity_threshold = similarity_threshold
        self.min_text_confidence = 60  # Minimum confidence for OCR

    def _is_similar(self, frame1: np.ndarray, frame2: np.ndarray) -> bool:
        """Check if frames are too similar"""
        hist1 = cv2.calcHist([frame1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([frame2], [0], None, [256], [0, 256])
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return similarity > self.similarity_threshold

    def _is_relevant(self, frame: np.ndarray) -> bool:
        """Check if frame contains useful content"""
        # Check if frame is not too dark
        if np.mean(frame) < 30:
            return False

        # Check if frame has enough edges (content)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        if np.count_nonzero(edges) < 1000:
            return False

        return True

    def extract_frames(self, video_path: str) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        frames = []
        prev_frame = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if self._is_relevant(frame):
                if prev_frame is None or not self._is_similar(frame, prev_frame):
                    frames.append(frame)
                    prev_frame = frame

        cap.release()
        return frames

    def process_frames(self, frames: List[np.ndarray]) -> str:
        texts = []
        for frame in frames:
            # Preprocess for OCR
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Apply thresholding
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Get OCR data including confidence scores
            data = pytesseract.image_to_data(
                thresh,
                output_type=pytesseract.Output.DICT
            )

            frame_text = []
            # Filter text by confidence
            for i, conf in enumerate(data['conf']):
                if conf > self.min_text_confidence:
                    text = data['text'][i].strip()
                    if text:
                        frame_text.append(text)

            if frame_text:
                texts.append(' '.join(frame_text))

        return '\n'.join(texts)
