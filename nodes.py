import os
import tempfile
from PIL import Image as PILImage
import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image, to_tensor

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
        # Debugging Information
        print("DEBUG: Received image type:", type(image))

        # Convert Torch Tensor to PIL Image
        if isinstance(image, torch.Tensor):
            print("DEBUG: original image shape:", image.shape)

            # Remove batch dimension if present (from [1, H, W, C] or [1, C, H, W])
            if image.ndim == 4:
                image = image.squeeze(0)
                print("DEBUG: after removing batch, shape:", image.shape)

            # If image is in (H, W, C) format, convert it to PIL
            if image.shape[0] in [1, 3]:  # (C, H, W) format
                pil_image = to_pil_image(image)
            else:  # Already in (H, W, C) format
                pil_image = PILImage.fromarray((image.numpy() * 255).astype(np.uint8))
        else:
            # Assume input is already a PIL image
            pil_image = image.convert("RGB")

        # Apply Brush Strokes Using the Chosen Method
        if method == "imagick":
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
            os.remove(in_path)
            os.remove(out_path)

        elif method == "opencv":
            if cv2 is None:
                raise RuntimeError("OpenCV not installed or failed to import.")

            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            sigma_s = float(strength) * 12  
            sigma_r = 0.45  
            stylized = cv2.stylization(cv_image, sigma_s=sigma_s, sigma_r=sigma_r)
            processed_pil = PILImage.fromarray(cv2.cvtColor(stylized, cv2.COLOR_BGR2RGB))
        else:
            raise ValueError("Unknown method. Choose either 'imagick' or 'opencv'.")

        # Convert Processed Image to Tensor
        result_tensor = to_tensor(processed_pil).unsqueeze(0)  # Convert to [1, C, H, W]
        print("DEBUG: result_tensor shape:", result_tensor.shape)

        # Ensure Correct Data Type
        if result_tensor.dtype != torch.uint8:
            result_tensor = (result_tensor * 255).clamp(0, 255).to(torch.uint8)

        return (result_tensor,)
