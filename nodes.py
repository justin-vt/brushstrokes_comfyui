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
    from torchvision.transforms.functional import to_pil_image, to_tensor
except ImportError:
    torch = None

class BrushStrokesNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Expecting a torch.Tensor in NHWC format: [1, H, W, C]
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
            # Remove batch dimension (assuming shape [1, H, W, C])
            if image.ndim == 4:
                image = image[0]
                print("DEBUG: after removing batch, shape:", image.shape)
            # Now image is assumed to be in HWC order.
            # to_pil_image expects a tensor in CHW, so permute HWC -> CHW.
            pil_image = to_pil_image(image.permute(2, 0, 1).cpu())
        else:
            # Otherwise, assume it's already a PIL image.
            pil_image = image.convert("RGB")
        
        # Process the image using the chosen method.
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
            processed_pil = PILImage.open(out_path).convert("RGB")
            if os.path.exists(in_path):
                os.remove(in_path)
            if os.path.exists(out_path):
                os.remove(out_path)
        elif method == "opencv":
            if cv2 is None:
                raise RuntimeError("OpenCV not installed or failed to import.")
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            sigma_s = float(strength) * 12  # Mapping can be adjusted as needed.
            sigma_r = 0.45  # Fixed parameter.
            stylized = cv2.stylization(cv_image, sigma_s=sigma_s, sigma_r=sigma_r)
            processed_pil = PILImage.fromarray(cv2.cvtColor(stylized, cv2.COLOR_BGR2RGB))
        else:
            raise ValueError("Unknown method. Choose either 'imagick' or 'opencv'.")

        # Convert the processed PIL image back to a torch.Tensor in NHWC format.
        # to_tensor returns a tensor in CHW order.
        result_tensor = to_tensor(processed_pil)
        # Permute from CHW to HWC.
        result_tensor = result_tensor.permute(1, 2, 0)
        # Add batch dimension -> [1, H, W, C]
        result_tensor = result_tensor.unsqueeze(0)
        
        return {"image": result_tensor}
