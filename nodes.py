import os
import tempfile
from PIL import Image as PILImage

try:
    from wand.image import Image as WandImage
except ImportError:
    WandImage = None

try:
    from gmic import Gmic
except ImportError:
    Gmic = None

class BrushStrokesNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pixels": ("IMAGE",),  # Accepts dict/PIL image/torch.Tensor
                "method": (["imagick", "gmic"], {"default": "imagick", "label": "Which method to use"}),
                "style": (["oilpaint", "paint", "painting", "brushify"], {"default": "oilpaint", "label": "Style"}),
                "strength": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 1.0, "label": "Strength"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_brush_strokes"
    CATEGORY = "Custom/Artistic"

    def apply_brush_strokes(self, pixels, method, style, strength):
        # Attempt to import torch and torchvision transforms for tensor conversion.
        try:
            import torch
            from torchvision.transforms.functional import to_pil_image
        except ImportError:
            torch = None

        # Determine input type:
        pil_image = None

        if isinstance(pixels, dict):
            # Check for common keys ("image" or "IMAGE")
            if "image" in pixels:
                pil_image = pixels["image"]
            elif "IMAGE" in pixels:
                pil_image = pixels["IMAGE"]
            else:
                raise ValueError("Input must be a dict with a PIL image under the key 'image' or a PIL image directly")
        elif isinstance(pixels, PILImage.Image):
            pil_image = pixels
        elif torch is not None and isinstance(pixels, torch.Tensor):
            pil_image = pixels
        else:
            raise ValueError("Input must be a dict with a PIL image under the key 'image' or a PIL image directly")

        # If pil_image is a torch.Tensor, convert it to a PIL image.
        if torch is not None and isinstance(pil_image, torch.Tensor):
            # If there is a batch dimension, take the first image.
            if pil_image.ndim == 4:
                pil_image = pil_image[0]
            # If there are more than 4 channels (unexpected), slice to first 3 channels.
            if pil_image.ndim == 3 and pil_image.shape[0] > 4:
                pil_image = pil_image[:3, :, :]
            pil_image = to_pil_image(pil_image.cpu())
        # If it's not a PIL image at this point, try converting.
        elif not isinstance(pil_image, PILImage.Image):
            raise ValueError("Input must be a dict with a PIL image under the key 'image' or a PIL image directly")

        # Convert to RGB (if not already)
        pil_image = pil_image.convert("RGB")

        # Save the PIL image to a temporary file for processing.
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in:
            in_path = tmp_in.name
            pil_image.save(in_path, "PNG")

        out_path = in_path + "_out.png"

        # Apply brush effect using the selected method.
        if method == "imagick":
            if WandImage is None:
                raise RuntimeError("Wand (ImageMagick) not installed or failed to import.")
            with WandImage(filename=in_path) as wand_img:
                if style == "oilpaint":
                    wand_img.oil_paint(radius=float(strength))
                elif style == "paint":
                    wand_img.oil_paint(radius=float(strength) / 2.0)
                else:
                    wand_img.oil_paint(radius=float(strength))
                wand_img.save(filename=out_path)
        elif method == "gmic":
            if Gmic is None:
                raise RuntimeError("gmic-py not installed or failed to import.")
            g = Gmic()
            g.run(in_path)
            if style == "painting":
                cmd = f"fx_painting {strength},0.5,0.4,0,0,0"
            elif style == "brushify":
                cmd = f"fx_brushify {strength},50,0.5,0.7,0"
            else:
                cmd = f"fx_painting {strength},0.5,0.4,0,0,0"
            g.run(cmd)
            g.run(out_path)

        # Open and convert the processed image.
        result_img = PILImage.open(out_path).convert("RGB")

        # Clean up temporary files.
        if os.path.exists(in_path):
            os.remove(in_path)
        if os.path.exists(out_path):
            os.remove(out_path)

        return ({"image": result_img},)
