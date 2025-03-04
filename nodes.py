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
                "pixels": ("IMAGE",),  # Accepts a dict with "image" key or a PIL image directly
                "method": (["imagick", "gmic"], {"default": "imagick", "label": "Which method to use"}),
                "style": (["oilpaint", "paint", "painting", "brushify"], {"default": "oilpaint", "label": "Style"}),
                "strength": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 1.0, "label": "Strength"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_brush_strokes"
    CATEGORY = "Custom/Artistic"

    def apply_brush_strokes(self, pixels, method, style, strength):
        # Accept either a dict containing a PIL image or a raw PIL image
        if isinstance(pixels, dict) and "image" in pixels:
            pil_image = pixels["image"].convert("RGB")
        elif isinstance(pixels, PILImage.Image):
            pil_image = pixels.convert("RGB")
        else:
            raise ValueError("Input must be a dict with a PIL image under the key 'image' or a PIL image directly.")

        # Save the PIL image to a temporary file so that it can be processed by Wand or G'MIC.
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in:
            in_path = tmp_in.name
            pil_image.save(in_path, "PNG")

        out_path = in_path + "_out.png"

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

        result_img = PILImage.open(out_path).convert("RGB")

        if os.path.exists(in_path):
            os.remove(in_path)
        if os.path.exists(out_path):
            os.remove(out_path)

        return ({"image": result_img},)
