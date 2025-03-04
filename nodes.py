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
                "pixels": ("IMAGE",),
                "method": (["imagick", "gmic"], {"default": "imagick"}),
                "style": (["oilpaint", "paint", "painting", "brushify"], {"default": "oilpaint"}),
                "strength": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 1.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_brush_strokes"
    CATEGORY = "Custom/Artistic"

    def apply_brush_strokes(self, pixels, method, style, strength):
        if "image" not in pixels:
            return (pixels,)

        pil_image = pixels["image"].convert("RGB")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in:
            in_path = tmp_in.name
            pil_image.save(in_path, "PNG")

        out_path = in_path + "_out.png"

        if method == "imagick":
            if WandImage is None:
                raise RuntimeError("Wand (ImageMagick) not installed.")
            with WandImage(filename=in_path) as wand_img:
                if style == "oilpaint":
                    wand_img.oil_paint(radius=float(strength))
                wand_img.save(filename=out_path)

        elif method == "gmic":
            if Gmic is None:
                raise RuntimeError("gmic-py not installed.")
            g = Gmic()
            g.run(in_path)
            cmd = f"fx_painting {strength},0.5,0.4,0,0,0"
            g.run(cmd)
            g.run(out_path)

        result_img = PILImage.open(out_path).convert("RGB")

        os.remove(in_path)
        os.remove(out_path)

        return ({"image": result_img},)
