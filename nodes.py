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

try:
    import torch
    from torchvision.transforms.functional import to_pil_image
except ImportError:
    torch = None

class BrushStrokesNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
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
        # Debug: print the type and shape (if tensor) of the input.
        print("DEBUG: type(image):", type(image))
        if torch is not None and isinstance(image, torch.Tensor):
            print("DEBUG: original image shape:", image.shape)
            # If the tensor has a batch dimension, select the first image.
            if image.ndim == 4:
                image = image[0]
                print("DEBUG: using first element from batch, new shape:", image.shape)
            # At this point, if the tensor is 3D, check if it appears to be in CHW format.
            # If the first dimension (channels) is not 1, 3, or 4 but the last dimension is not in that set either,
            # then it's likely in CHW format. Convert it to HWC.
            if image.ndim == 3:
                if image.shape[0] not in [1, 3, 4] and image.shape[-1] not in [1, 3, 4]:
                    print("DEBUG: Permuting tensor from CHW to HWC")
                    image = image.permute(1, 2, 0)
                    print("DEBUG: new shape after permutation:", image.shape)
            pil_image = to_pil_image(image.cpu())
        else:
            # Assume the input is already a PIL image.
            pil_image = image.convert("RGB")
        
        if method == "imagick":
            if WandImage is None:
                raise RuntimeError("Wand (ImageMagick) not installed or failed to import.")
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
            if os.path.exists(in_path):
                os.remove(in_path)
            if os.path.exists(out_path):
                os.remove(out_path)
        elif method == "opencv":
            if cv2 is None:
                raise RuntimeError("OpenCV not installed or failed to import.")
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            sigma_s = float(strength) * 12  # Adjust mapping as needed.
            sigma_r = 0.45  # Fixed parameter.
            stylized = cv2.stylization(cv_image, sigma_s=sigma_s, sigma_r=sigma_r)
            result_img = PILImage.fromarray(cv2.cvtColor(stylized, cv2.COLOR_BGR2RGB))
        else:
            raise ValueError("Unknown method. Choose either 'imagick' or 'opencv'.")
        
        return {"image": result_img}
