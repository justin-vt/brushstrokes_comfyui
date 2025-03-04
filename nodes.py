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
                # Typically from LoadImage: a 4D torch.Tensor in either [1,H,W,3] or [1,3,H,W].
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
        # If it's already a torch.Tensor, figure out shape.
        # ComfyUI typically gives shape [1,H,W,3] or [1,3,H,W].
        if torch is not None and isinstance(image, torch.Tensor):
            print("DEBUG: original image shape:", image.shape)

            # Remove the batch dimension (shape[0] == 1).
            if image.ndim == 4 and image.shape[0] == 1:
                image = image[0]
                # Now possible shapes are [H,W,3] or [3,H,W].
                if image.ndim == 3:
                    # Channel-last if last dim == 3 or 4, channel-first if first dim == 3 or 4.
                    if image.shape[-1] == 3 or image.shape[-1] == 4:
                        # channels-last: [H,W,3/4]
                        image = image.permute(2, 0, 1)  # => [3/4, H, W]
                    # else if image.shape[0] is 3 or 4, it's already channels-first
                else:
                    raise ValueError("Unsupported shape after removing batch dimension: {}".format(image.shape))

                print("DEBUG: after removing batch, shape (channels-first):", image.shape)
                pil_image = to_pil_image(image.cpu())
            else:
                # Fallback if somehow it's not 4D or not batch size 1, just try to_pil_image
                pil_image = to_pil_image(image)
        else:
            # If it’s not a torch.Tensor, assume it’s already a PIL image
            pil_image = image.convert("RGB")

        # Apply the selected style with Imagick/Wand or OpenCV
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
            sigma_s = float(strength) * 12
            sigma_r = 0.45  # tweak if desired
            stylized = cv2.stylization(cv_image, sigma_s=sigma_s, sigma_r=sigma_r)
            processed_pil = PILImage.fromarray(cv2.cvtColor(stylized, cv2.COLOR_BGR2RGB))
        else:
            raise ValueError("Unknown method. Choose either 'imagick' or 'opencv'.")

        if processed_pil is None:
            raise RuntimeError("Processed image is None.")

        # Convert processed image back to a [1, 3, H, W] torch.Tensor.
        result_tensor = to_tensor(processed_pil)    # => [3,H,W]
        result_tensor = result_tensor.unsqueeze(0)  # => [1,3,H,W]

        print("DEBUG: result_tensor shape:", result_tensor.shape)
        print("DEBUG: result_tensor dtype:", result_tensor.dtype)

        return (result_tensor,)
