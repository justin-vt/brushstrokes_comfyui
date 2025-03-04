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
    np = None

class BrushStrokesNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Expect an image from the Load Image node. It can be:
                # - a dict with key "image" or "IMAGE",
                # - a raw PIL image, or
                # - a numpy array.
                "image": ("IMAGE",),
                "method": (["imagick", "opencv"], {"default": "imagick", "label": "Which method to use"}),
                "style": (["oilpaint", "paint"], {"default": "oilpaint", "label": "Style"}),
                "strength": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 1.0, "label": "Strength"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_brush_strokes"
    CATEGORY = "Custom/Artistic"

    def apply_brush_strokes(self, image, method, style, strength):
        # If the input is a list (common in node linking), take the first element.
        if isinstance(image, list):
            image = image[0]

        # Determine the input image.
        if isinstance(image, dict):
            if "image" in image:
                pil_image = image["image"]
            elif "IMAGE" in image:
                pil_image = image["IMAGE"]
            else:
                raise ValueError("Input must be a dict with a PIL image under the key 'image' or a PIL image directly")
        elif isinstance(image, PILImage.Image):
            pil_image = image
        elif np is not None and isinstance(image, np.ndarray):
            pil_image = PILImage.fromarray(image)
        else:
            raise ValueError("Input must be a dict with a PIL image under the key 'image' or a PIL image directly")

        # Ensure the image is in RGB.
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
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            # Map 'strength' to OpenCV's stylization sigma_s parameter.
            sigma_s = float(strength) * 12  # Adjust this mapping as needed.
            sigma_r = 0.45  # Fixed range parameter.
            stylized = cv2.stylization(cv_image, sigma_s=sigma_s, sigma_r=sigma_r)
            result_img = PILImage.fromarray(cv2.cvtColor(stylized, cv2.COLOR_BGR2RGB))
        else:
            raise ValueError("Unknown method. Choose either 'imagick' or 'opencv'.")

        # Return the processed image as a dict with the key "image".
        return {"image": result_img}
