from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import torch
import io
from wand.image import Image as WandImage
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

from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import torch
import io
from wand.image import Image as WandImage
import cv2

###############################################
# 1. WandBrushStrokesNode (using Wand/ImageMagick)
###############################################
class WandBrushStrokesNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": (
                    "IMAGE",
                    {"tooltip": "Input image tensor from the Load Image node."}
                ),
                "effect": (
                    ["Swirl", "Wave", "Pixelation"],
                    {"tooltip": (
                        "Select the Wand effect to apply.\n"
                        "Suggested defaults:\n"
                        "Expressionist: Swirl (degree ≈90°)\n"
                        "Realist: Wave (amplitude ≈3, wavelength ≈80)\n"
                        "Abstract: Pixelation (pixel_size ≈15)\n"
                        "Impressionist: Swirl (degree ≈30°)"
                    )}
                ),
                "degree": (
                    "FLOAT",
                    {"default": 30.0, "min": 0.0, "max": 360.0, "step": 1.0,
                     "tooltip": "Swirl: swirl degree in degrees (only used if effect is Swirl)."}
                ),
                "amplitude": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.0, "max": 100.0, "step": 1.0,
                     "tooltip": "Wave: amplitude in pixels (only used if effect is Wave)."}
                ),
                "wavelength": (
                    "FLOAT",
                    {"default": 50.0, "min": 1.0, "max": 200.0, "step": 1.0,
                     "tooltip": "Wave: wavelength in pixels (only used if effect is Wave)."}
                ),
                "pixel_size": (
                    "FLOAT",
                    {"default": 10.0, "min": 1.0, "max": 50.0, "step": 1.0,
                     "tooltip": "Pixelation: size factor for pixelation (only used if effect is Pixelation)."}
                ),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "Utility"
    
    def process_image(self, input_image: torch.Tensor, effect: str, degree: float, amplitude: float, wavelength: float, pixel_size: float):
        # Convert input tensor (shape [1, H, W, 3] with values in [0,1]) to a uint8 numpy array.
        img_np = input_image[0].cpu().numpy()
        img_uint8 = (np.clip(img_np * 255.0, 0, 255)).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        buf.seek(0)
        with WandImage(blob=buf.getvalue()) as wand_img:
            if effect == "Swirl":
                wand_img.swirl(degree=degree)
            elif effect == "Wave":
                # Wave effect uses positional arguments: amplitude, wavelength
                wand_img.wave(amplitude, wavelength)
            elif effect == "Pixelation":
                orig_width, orig_height = wand_img.width, wand_img.height
                new_width = max(1, int(orig_width / pixel_size))
                new_height = max(1, int(orig_height / pixel_size))
                wand_img.resize(new_width, new_height)
                wand_img.resize(orig_width, orig_height)
            processed_blob = wand_img.make_blob(format='png')
        processed_pil = Image.open(io.BytesIO(processed_blob)).convert("RGB")
        processed_np = np.array(processed_pil).astype(np.float32) / 255.0
        output_tensor = torch.from_numpy(processed_np)[None, ...]
        return (output_tensor,)

###############################################
# 2. PILBrushStrokesNode (using native PIL enhancements)
###############################################
class PILBrushStrokesNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": (
                    "IMAGE",
                    {"tooltip": "Input image tensor from the Load Image node."}
                ),
                "effect": (
                    ["Color Painting", "Realistic", "Vibrance"],
                    {"tooltip": (
                        "Select the PIL effect to apply.\n"
                        "Suggested defaults:\n"
                        "Expressionist: Color Painting (strength ≈30, radius ≈2)\n"
                        "Realist: Realistic (strength ≈10, radius ≈10)\n"
                        "Abstract: Vibrance (strength ≈40, radius ≈12)\n"
                        "Impressionist: Color Painting (strength ≈20, radius ≈5)"
                    )}
                ),
                "strength": (
                    "FLOAT",
                    {"default": 10.0, "min": 0.0, "max": 100.0, "step": 1.0,
                     "tooltip": "Controls effect intensity (e.g. saturation/brightness factor); used only with the selected PIL effect."}
                ),
                "radius": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.0, "max": 100.0, "step": 1.0,
                     "tooltip": "Modulates posterization depth or contrast; used only with the selected PIL effect."}
                ),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "Utility"
    
    def process_image(self, input_image: torch.Tensor, effect: str, strength: float, radius: float):
        # Convert input tensor to uint8 numpy array.
        img_np = input_image[0].cpu().numpy()
        img_uint8 = (np.clip(img_np * 255.0, 0, 255)).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8).convert("RGB")
        if effect == "Color Painting":
            # Increase saturation aggressively.
            enhancer = ImageEnhance.Color(pil_img)
            factor = 1 + (strength / 10.0)
            pil_img = enhancer.enhance(factor)
            # Apply posterization: fewer bits = stronger effect.
            bits = max(1, min(8, int(8 - (radius / 10.0))))
            pil_img = ImageOps.posterize(pil_img, bits)
        elif effect == "Realistic":
            # Subtle brightness and contrast adjustment.
            enhancer = ImageEnhance.Brightness(pil_img)
            pil_img = enhancer.enhance(1 + (strength / 20.0))
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(1 + (radius / 15.0))
        elif effect == "Vibrance":
            # Stronger saturation and contrast for a vivid, abstract look.
            enhancer = ImageEnhance.Color(pil_img)
            pil_img = enhancer.enhance(1 + (strength / 8.0))
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(1 + (radius / 15.0))
        processed_np = np.array(pil_img).astype(np.float32) / 255.0
        output_tensor = torch.from_numpy(processed_np)[None, ...]
        return (output_tensor,)

NODE_CLASS_MAPPINGS = {
    "OpenCVBrushStrokesNode": OpenCVBrushStrokesNode,
    "WandBrushStrokesNode": WandBrushStrokesNode,
    "PILBrushStrokesNode": PILBrushStrokesNode
}
