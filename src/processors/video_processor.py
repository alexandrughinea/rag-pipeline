from typing import List

import cv2
import numpy as np
from pytesseract import pytesseract


class VideoProcessor:
    def __init__(self,
                 sample_rate: int = 1,
                 similarity_threshold: float = 0.9,
                 max_frames: int = 100,
                 max_dimension: int = 1980,
                 min_text_confidence: int = 60):
        self.sample_rate = sample_rate
        self.similarity_threshold = similarity_threshold
        self.max_frames = max_frames
        self.max_dimension = max_dimension
        self.min_text_confidence = min_text_confidence

    def _is_similar(self, frame1: np.ndarray, frame2: np.ndarray) -> bool:
        """Check if frames are too similar"""
        hist1 = cv2.calcHist([frame1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([frame2], [0], None, [256], [0, 256])
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return similarity > self.similarity_threshold

    def _is_relevant(self, frame: np.ndarray) -> bool:
        """Check if frame contains useful content"""
        if np.mean(frame) < 30:  # Too dark
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        if np.count_nonzero(edges) < 1000:  # Not enough edges/content
            return False

        return True

    def extract_frames(self, video_path: str) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps

        # Calculate frame interval
        frame_interval = max(1, total_frames // self.max_frames)

        print(f"Video stats - FPS: {fps}, Frames: {total_frames}, Duration: {duration:.2f}s")

        frames = []
        prev_frame = None
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process only every nth frame
            if frame_count % frame_interval != 0:
                frame_count += 1
                continue

            # Resize if too large
            height, width = frame.shape[:2]
            if width > self.max_dimension or height > self.max_dimension:
                scale = self.max_dimension / max(width, height)
                frame = cv2.resize(frame, None, fx=scale, fy=scale)

            # Check relevance and similarity
            if self._is_relevant(frame):
                if prev_frame is None or not self._is_similar(frame, prev_frame):
                    frames.append(frame)
                    prev_frame = frame

            if len(frames) >= self.max_frames:
                break

            frame_count += 1

        cap.release()
        print(f"Extracted {len(frames)} unique and relevant frames")
        return frames

    def process_frames(self, frames: List[np.ndarray]) -> str:
        texts = []
        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray)

            # Improve contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)

            # Get OCR data
            data = pytesseract.image_to_data(enhanced, output_type=pytesseract.Output.DICT)

            # Skip frames with no text
            if all(text == '' for text in data['text']):
                print(f"Skipping frame {i} - no text detected")
                continue

            frame_text = []
            for j, conf in enumerate(data['conf']):
                if conf > self.min_text_confidence:
                    text = data['text'][j].strip()
                    if text and len(text) > 1:  # Skip single characters
                        frame_text.append(text)

            if frame_text:  # Only add frames that have valid text
                texts.append(' '.join(frame_text))

        return '\n'.join(texts)

