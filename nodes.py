import os
import tempfile
from PIL import Image as PILImage

try:
    from wand.image import Image as WandImage
except ImportError:
    WandImage = None

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None

class BrushStrokesNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Accepts a dict with key "image" or "IMAGE", a raw PIL image,
                # or even a list of such images.
                "pixels": ("IMAGE",),
                "method": (["imagick", "opencv"], {"default": "imagick", "label": "Which method to use"}),
                "style": (["oilpaint", "paint"], {"default": "oilpaint", "label": "Style"}),
                "strength": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 1.0, "label": "Strength"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_brush_strokes"
    CATEGORY = "Custom/Artistic"

    def apply_brush_strokes(self, pixels, method, style, strength):
        # If the input is a list, take the first element.
        if isinstance(pixels, list):
            pixels = pixels[0]

        # Determine the input image.
        if isinstance(pixels, dict):
            if "image" in pixels:
                pil_image = pixels["image"]
            elif "IMAGE" in pixels:
                pil_image = pixels["IMAGE"]
            else:
                raise ValueError("Input must be a dict with a PIL image under the key 'image' or a PIL image directly")
        elif isinstance(pixels, PILImage.Image):
            pil_image = pixels
        else:
            raise ValueError("Input must be a dict with a PIL image under the key 'image' or a PIL image directly")

        # Ensure the image is in RGB mode.
        pil_image = pil_image.convert("RGB")

        if method == "imagick":
            if WandImage is None:
                raise RuntimeError("Wand (ImageMagick) not installed or failed to import.")
            # Save the image to a temporary file.
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in:
                in_path = tmp_in.name
                pil_image.save(in_path, "PNG")
            out_path = in_path + "_out.png"
            with WandImage(filename=in_path) as wand_img:
                if style == "oilpaint":
                    wand_img.oil_paint(radius=float(strength))
                elif style == "paint":
                    wand_img.oil_paint(radius=float(strength) / 2.0)
                else:
                    wand_img.oil_paint(radius=float(strength))
                wand_img.save(filename=out_path)
            result_img = PILImage.open(out_path).convert("RGB")
            # Clean up temporary files.
            if os.path.exists(in_path):
                os.remove(in_path)
            if os.path.exists(out_path):
                os.remove(out_path)
        elif method == "opencv":
            if cv2 is None:
                raise RuntimeError("OpenCV not installed or failed to import.")
            # Convert the PIL image to a NumPy array in BGR format.
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            # Map 'strength' to a spatial parameter (sigma_s); adjust as needed.
            sigma_s = float(strength) * 12
            sigma_r = 0.45  # fixed range parameter for stylization
            stylized = cv2.stylization(cv_image, sigma_s=sigma_s, sigma_r=sigma_r)
            result_img = PILImage.fromarray(cv2.cvtColor(stylized, cv2.COLOR_BGR2RGB))
        else:
            raise ValueError("Unknown method. Choose either 'imagick' or 'opencv'.")

        return ({"image": result_img},)
