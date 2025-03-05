from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import torch
import cv2
import io
from wand.image import Image as WandImage

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
                    {
                        "tooltip": "Input image tensor (shape [1,H,W,3], values in [0,1]) from the Load Image node."
                    }
                ),
                "effect": (
                    ["Swirl", "Wave", "Pixelation"],
                    {"tooltip": (
                        "Choose the Wand effect to simulate brush stroke texture.\n"
                        "• Swirl: Applies a rotation distortion. Recommended:\n"
                        "    - Expressionist: degree ≈ 10° (subtle local twist)\n"
                        "    - Realist:       degree ≈ 5° (very minimal twist)\n"
                        "    - Impressionist: degree ≈ 30° (noticeable, but still at brush stroke level)\n"
                        "• Wave: Applies a ripple distortion. Recommended for Realist only:\n"
                        "    - amplitude ≈ 1–2 pixels, wavelength ≈ 100 pixels\n"
                        "• Pixelation: Flattens detail. Recommended for Abstract:\n"
                        "    - pixel_size ≈ 15\n"
                        "Parameters not used by the selected effect will be ignored."
                    )}
                ),
                "degree": (
                    "FLOAT",
                    {"default": 10.0, "min": 0.0, "max": 360.0, "step": 1.0,
                     "tooltip": (
                         "Swirl degree in degrees. Only used when 'Swirl' effect is selected.\n"
                         "Use ~10° for Expressionist, ~5° for Realist, ~30° for Impressionist."
                     )}
                ),
                "amplitude": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.5,
                     "tooltip": (
                         "Wave amplitude in pixels. Only used when 'Wave' effect is selected.\n"
                         "Use a very low value (≈1–2) so the ripple is subtle."
                     )}
                ),
                "wavelength": (
                    "FLOAT",
                    {"default": 100.0, "min": 1.0, "max": 300.0, "step": 1.0,
                     "tooltip": (
                         "Wave wavelength in pixels. Only used when 'Wave' effect is selected.\n"
                         "A higher value (≈100) produces a gentle, localized ripple."
                     )}
                ),
                "pixel_size": (
                    "FLOAT",
                    {"default": 10.0, "min": 1.0, "max": 50.0, "step": 1.0,
                     "tooltip": (
                         "Pixelation size factor. Only used when 'Pixelation' effect is selected.\n"
                         "For an Abstract effect, try a value around 15 to flatten details."
                     )}
                ),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "Utility"
    
    def process_image(self, input_image: torch.Tensor, effect: str, degree: float, amplitude: float, wavelength: float, pixel_size: float):
        # Convert input tensor (normalized [0,1]) to a uint8 numpy array.
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
                # Use positional arguments for wave: amplitude and wavelength.
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
                    {
                        "tooltip": "Input image tensor (shape [1,H,W,3], normalized to [0,1]) from the Load Image node."
                    }
                ),
                "effect": (
                    ["Color Painting", "Realistic", "Vibrance"],
                    {"tooltip": (
                        "Select the PIL effect to simulate painterly brush strokes:\n"
                        "• Color Painting: Increases saturation to mimic vivid, textured strokes. Suggested:\n"
                        "    - Expressionist: strength ≈ 30, radius ≈ 2 (strong saturation, fine posterization)\n"
                        "    - Impressionist: strength ≈ 20, radius ≈ 5 (moderate, painterly look)\n"
                        "• Realistic: Adjusts brightness/contrast subtly for a natural appearance. Suggested: strength ≈ 10, radius ≈ 10\n"
                        "• Vibrance: Boosts saturation and contrast for a bold, graphic style. Suggested for Abstract: strength ≈ 40, radius ≈ 12\n"
                        "Parameters not used by the selected effect will be ignored."
                    )}
                ),
                "strength": (
                    "FLOAT",
                    {"default": 10.0, "min": 0.0, "max": 100.0, "step": 1.0,
                     "tooltip": (
                         "Effect intensity (e.g., saturation/brightness factor).\n"
                         "Only used with the selected PIL effect. For Color Painting, higher values increase saturation."
                     )}
                ),
                "radius": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.0, "max": 100.0, "step": 1.0,
                     "tooltip": (
                         "Modulates posterization depth or contrast adjustment.\n"
                         "Only used with the selected PIL effect. Lower values produce more discrete color bands."
                     )}
                ),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "Utility"
    
    def process_image(self, input_image: torch.Tensor, effect: str, strength: float, radius: float):
        img_np = input_image[0].cpu().numpy()
        img_uint8 = (np.clip(img_np * 255.0, 0, 255)).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8).convert("RGB")
        if effect == "Color Painting":
            enhancer = ImageEnhance.Color(pil_img)
            factor = 1 + (strength / 10.0)  # Increase saturation
            pil_img = enhancer.enhance(factor)
            bits = max(1, min(8, int(8 - (radius / 10.0))))
            pil_img = ImageOps.posterize(pil_img, bits)
        elif effect == "Realistic":
            enhancer = ImageEnhance.Brightness(pil_img)
            pil_img = enhancer.enhance(1 + (strength / 20.0))
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(1 + (radius / 15.0))
        elif effect == "Vibrance":
            enhancer = ImageEnhance.Color(pil_img)
            pil_img = enhancer.enhance(1 + (strength / 8.0))
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(1 + (radius / 15.0))
        processed_np = np.array(pil_img).astype(np.float32) / 255.0
        output_tensor = torch.from_numpy(processed_np)[None, ...]
        return (output_tensor,)

class OpenCVBrushStrokesNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": (
                    "IMAGE",
                    {
                        "tooltip": "Input image tensor (shape [1,H,W,3], normalized to [0,1]) from the Load Image node."
                    }
                ),
                "effect": (
                    ["Stylization", "Edge Preserving"],
                    {"tooltip": (
                        "Select the OpenCV effect to simulate painterly brush strokes:\n"
                        " - Stylization: Creates an abstract, painterly look.\n"
                        "   Suggested presets:\n"
                        "      Expressionist: sigma_s≈100, sigma_r≈0.50\n"
                        "      Abstract:      sigma_s≈150, sigma_r≈0.60\n"
                        "      Impressionist: sigma_s≈90,  sigma_r≈0.55\n"
                        " - Edge Preserving: Smooths the image while keeping prominent edges.\n"
                        "   Suggested preset:\n"
                        "      Realist:       sigma_s≈80,  sigma_r≈0.40 (filter_flag=1)\n"
                        "Parameters not used by the selected effect will be ignored."
                    )}
                ),
                "sigma_s": (
                    "FLOAT",
                    {"default": 60.0, "min": 0.0, "max": 200.0, "step": 1.0,
                     "tooltip": (
                         "Spatial scale (sigma_s) that controls the size of the smoothing effect.\n"
                         "Higher values (e.g. 100 for Expressionist, 150 for Abstract) yield larger, smoother areas.\n"
                         "Used with the selected effect."
                     )}
                ),
                "sigma_r": (
                    "FLOAT",
                    {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01,
                     "tooltip": (
                         "Range value (sigma_r) that determines how much variation between pixels is preserved.\n"
                         "Lower values smooth more aggressively (e.g. 0.50 for Expressionist, 0.55 for Impressionist, 0.40 for Realist).\n"
                         "Used with the selected effect."
                     )}
                ),
                "filter_flag": (
                    "INT",
                    {"default": 1, "min": 0, "max": 1, "step": 1,
                     "tooltip": (
                         "Filter flag for Edge Preserving effect: set to 0 for Normal or 1 for Recursive filtering.\n"
                         "Only used when effect is 'Edge Preserving' (suggested Realist preset: filter_flag=1)."
                     )}
                ),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "Utility"
    
    def process_image(self, input_image: torch.Tensor, effect: str, sigma_s: float, sigma_r: float, filter_flag: int):
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

###############################################
# Node Mappings
###############################################
NODE_CLASS_MAPPINGS = {
    "WandBrushStrokesNode": WandBrushStrokesNode,
    "PILBrushStrokesNode": PILBrushStrokesNode,
    "OpenCVBrushStrokesNode": OpenCVBrushStrokesNode
}
