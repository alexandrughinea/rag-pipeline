from typing import Optional, Tuple

from PIL import Image


class ImageProcessor:
    def __init__(self, min_dpi: int = 300, min_dimensions: Tuple[int, int] = (800, 600)):
        self.min_dpi = min_dpi
        self.min_width, self.min_height = min_dimensions

    def process_image(self, image_path: str, output_path: Optional[str] = None) -> Image.Image:
        """Process image by stripping metadata and ensuring minimum quality standards."""
        with Image.open(image_path) as img:
            # Create new image without metadata
            clean_img = Image.new(img.mode, img.size)
            clean_img.putdata(list(img.getdata()))

            # Check and adjust dimensions if needed
            if clean_img.width < self.min_width or clean_img.height < self.min_height:
                clean_img = self._resize_image(clean_img)

            # Save processed image
            output_path = output_path or image_path
            clean_img.save(output_path, quality=95)

            return img

    def _resize_image(self, img: Image.Image) -> Image.Image:
        """Resize image while maintaining aspect ratio."""
        width, height = img.size
        width_ratio = self.min_width / width if width < self.min_width else 1
        height_ratio = self.min_height / height if height < self.min_height else 1

        scale = max(width_ratio, height_ratio)

        if scale > 1:
            new_size = (
                int(width * scale),
                int(height * scale)
            )
            return img.resize(new_size, Image.Resampling.LANCZOS)
        return img