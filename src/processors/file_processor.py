import os

import pandas as pd
import pytesseract
from docx import Document
from PyPDF2 import PdfReader

from .image_processor import ImageProcessor
from .video_processor import VideoProcessor


class FileProcessor:
    def __init__(self):
        self.supported_formats = {
            # Raw text:
            ".txt": self._process_raw_text,
            ".md": self._process_raw_text,
            ".xml": self._process_raw_text,
            # Documents:
            ".csv": self._process_csv,
            ".pdf": self._process_pdf,
            ".docx": self._process_docx,
            ".doc": self._process_docx,
            # Images:
            ".png": self._process_image,
            ".jpg": self._process_image,
            ".jpeg": self._process_image,
            # Videos:
            ".avi": self._process_video,
            ".mp4": self._process_video,
            ".mov": self._process_video,
            ".mpeg": self._process_video,
        }

    def process_file(self, file_path):
        """Process a file and return extracted text."""
        ext = os.path.splitext(file_path)[1].lower()

        return self.supported_formats[ext](file_path)

    def _process_raw_text(self, file_path: str) -> str:
        """Process markup files (XML, Markdown) and extract text."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Error processing markup file: {str(e)}")

    def _process_csv(self, file_path) -> str:
        df = pd.read_csv(file_path)
        return df.to_string()

    def _process_pdf(self, file_path) -> str:
        text = ""
        with open(file_path, "rb") as file:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text

    def _process_docx(self, file_path) -> str:
        doc = Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])

    def _process_image(self, file_path) -> str:
        image_processor = ImageProcessor()
        processed_image = image_processor.process_image(file_path)

        return pytesseract.image_to_string(processed_image)

    def _process_video(self, file_path: str) -> str:
        # 1 frame per second for 30fps video
        video_processor = VideoProcessor(sample_rate=30)
        frames = video_processor.extract_frames(file_path)
        return video_processor.process_frames(frames)