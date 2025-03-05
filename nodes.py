from PIL import Image
import numpy as np
import torch
import cv2

class OpenCVBrushStrokesNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": (
                    "IMAGE",
                    {"tooltip": "Input image tensor from the Load Image node."}
                ),
                "effect": (
                    ["Stylization", "Edge Preserving"],
                    {"tooltip": (
                        "Select the OpenCV effect to apply.\n"
                        "Suggested defaults:\n"
                        "Expressionist: Stylization (sigma_s=100, sigma_r=0.50)\n"
                        "Realist: Edge Preserving (sigma_s=80, sigma_r=0.40)\n"
                        "Abstract: Stylization (sigma_s=150, sigma_r=0.60)\n"
                        "Impressionist: Stylization (sigma_s=90, sigma_r=0.55)"
                    )}
                ),
                "sigma_s": (
                    "FLOAT",
                    {"default": 60.0, "min": 0.0, "max": 200.0, "step": 1.0,
                     "tooltip": "Spatial scale (sigma_s) used for the effect."}
                ),
                "sigma_r": (
                    "FLOAT",
                    {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01,
                     "tooltip": "Range value (sigma_r) used for the effect."}
                ),
                "filter_flag": (
                    "INT",
                    {"default": 1, "min": 0, "max": 1, "step": 1,
                     "tooltip": "For Edge Preserving: 0 for Normal, 1 for Recursive filter."}
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "Utility"

    def process_image(self, input_image: torch.Tensor, effect: str, sigma_s: float, sigma_r: float, filter_flag: int):
        # Ensure a valid effect is selected.
        if effect not in ["Stylization", "Edge Preserving"]:
            effect = "Stylization"
        img_np = input_image[0].cpu().numpy()  # shape: [H, W, 3]
        img_uint8 = (np.clip(img_np * 255.0, 0, 255)).astype(np.uint8)
        cv_img = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        if effect == "Stylization":
            processed_cv = cv2.stylization(cv_img, sigma_s=sigma_s, sigma_r=sigma_r)
        elif effect == "Edge Preserving":
            processed_cv = cv2.edgePreservingFilter(cv_img, flags=filter_flag, sigma_s=sigma_s, sigma_r=sigma_r)
        else:
            processed_cv = cv_img
        processed_rgb = cv2.cvtColor(processed_cv, cv2.COLOR_BGR2RGB)
        processed_np = processed_rgb.astype(np.float32) / 255.0
        output_tensor = torch.from_numpy(processed_np)[None, ...]
        return (output_tensor,)

NODE_CLASS_MAPPINGS = {
    "OpenCVBrushStrokesNode": OpenCVBrushStrokesNode
}
