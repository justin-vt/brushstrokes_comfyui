import os
import tempfile
from PIL import Image as PILImage

# Import Wand for ImageMagick support.
try:
    from wand.image import Image as WandImage
except ImportError:
    WandImage = None

# Import OpenCV for alternative stylization.
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
                "pixels": ("IMAGE",),  # Accepts dict with "image" or "IMAGE" key, or a raw PIL image.
                "method": (["imagick", "opencv"], {"default": "imagick", "label": "Which method to use"}),
                "style": (["oilpaint", "paint"], {"default": "oilpaint", "label": "Style"}),
                "strength": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 1.0, "label": "Strength"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_brush_strokes"
    CATEGORY = "Custom/Artistic"

    def apply_brush_strokes(self, pixels, method, style, strength):
        # Handle input: accept dict with keys "image" or "IMAGE", or a raw PIL image.
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
        
        # Ensure image is in RGB.
        pil_image = pil_image.convert("RGB")
        
        if method == "imagick":
            if WandImage is None:
                raise RuntimeError("Wand (ImageMagick) not installed or failed to import.")
            # Save image to temporary file.
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in:
                in_path = tmp_in.name
                pil_image.save(in_path, "PNG")
            out_path = in_path + "_out.png"
            # Process image with Wand.
            with WandImage(filename=in_path) as wand_img:
                # Use oil_paint effect; adjust the radius using strength.
                if style == "oilpaint":
                    wand_img.oil_paint(radius=float(strength))
                elif style == "paint":
                    wand_img.oil_paint(radius=float(strength) / 2.0)
                else:
                    wand_img.oil_paint(radius=float(strength))
                wand_img.save(filename=out_path)
            # Load processed image.
            result_img = PILImage.open(out_path).convert("RGB")
            # Clean up temporary files.
            if os.path.exists(in_path):
                os.remove(in_path)
            if os.path.exists(out_path):
                os.remove(out_path)
        
        elif method == "opencv":
            if cv2 is None:
                raise RuntimeError("OpenCV not installed or failed to import.")
            # Convert PIL image to a numpy array in BGR format.
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            # OpenCV's stylization filter produces a painterly effect.
            # We'll map 'strength' (0 to 100) to sigma_s (spatial) parameter.
            # For example, if strength=5 then sigma_s ~ 60; adjust as needed.
            sigma_s = float(strength) * 12  # Adjust this mapping as desired.
            sigma_r = 0.45  # Fixed range parameter.
            stylized = cv2.stylization(cv_image, sigma_s=sigma_s, sigma_r=sigma_r)
            # Convert back from BGR to RGB and then to PIL image.
            result_img = PILImage.fromarray(cv2.cvtColor(stylized, cv2.COLOR_BGR2RGB))
        
        else:
            raise ValueError("Unknown method. Choose either 'imagick' or 'opencv'.")
        
        return ({"image": result_img},)
