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
                "method": (["imagick", "gmic"], {"default": "imagick", "label": "Which method to use"}),
                "style": (["oilpaint", "paint", "painting", "brushify"], {"default": "oilpaint", "label": "Style"}),
                "strength": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 1.0, "label": "Strength"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_brush_strokes"
    CATEGORY = "Custom/Artistic"

    def apply_brush_strokes(self, pixels, method, style, strength):
        # Determine input type: dict containing a PIL image or a torch.Tensor
        if isinstance(pixels, dict) and "image" in pixels:
            pil_image = pixels["image"].convert("RGB")
        else:
            # Assume it's a tensor; convert to PIL image using torchvision
            try:
                import torch
                from torchvision.transforms.functional import to_pil_image
            except ImportError:
                raise RuntimeError("torch and torchvision are required for tensor to PIL conversion.")
            if isinstance(pixels, torch.Tensor):
                pil_image = to_pil_image(pixels.cpu())
            else:
                raise ValueError("Unsupported image input type. Expected dict with 'image' key or a torch.Tensor.")

        # Save the PIL image to a temporary file so we can load it with Wand or G'MIC.
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
