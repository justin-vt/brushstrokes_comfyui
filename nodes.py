from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import torch
import io
from wand.image import Image as WandImage
import cv2

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
                "input_image": ("IMAGE", {"tooltip": "Input image tensor from the Load Image node."}),
                "preset": (["Custom", "Expressionist", "Realist", "Abstract", "Impressionist"], {"tooltip": "Select a painterly preset. Choosing a preset overrides manual parameters."}),
                # Only used in Custom mode.
                "effect": (["Swirl", "Wave", "Pixelation"], {"tooltip": "In Custom mode, choose the Wand effect to apply."}),
                "degree": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 360.0, "step": 1.0, "tooltip": "Swirl: the swirl degree (used when effect is Swirl or in Custom mode)."}),
                "amplitude": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 1.0, "tooltip": "Wave: amplitude in pixels (used when effect is Wave in Custom mode)."}),
                "wavelength": ("FLOAT", {"default": 50.0, "min": 1.0, "max": 200.0, "step": 1.0, "tooltip": "Wave: wavelength in pixels (used when effect is Wave in Custom mode)."}),
                "pixel_size": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 50.0, "step": 1.0, "tooltip": "Pixelation: size factor for pixelation (used when effect is Pixelation in Custom mode)."})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "Utility"
    
    def process_image(self, input_image: torch.Tensor, preset: str, effect: str, degree: float, amplitude: float, wavelength: float, pixel_size: float):
        # Presets override manual parameters.
        if preset != "Custom":
            if preset == "Expressionist":
                effect = "Swirl"
                degree = 45.0
            elif preset == "Realist":
                effect = "Wave"
                amplitude, wavelength = 8.0, 60.0
            elif preset == "Abstract":
                effect = "Pixelation"
                pixel_size = 12.0
            elif preset == "Impressionist":
                effect = "Swirl"
                degree = 30.0  # a lighter swirl
        # Convert input tensor to uint8 numpy array.
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
                wand_img.wave(amplitude=amplitude, wavelength=wavelength)
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
# 2. OpenCVBrushStrokesNode (using OpenCV)
###############################################
class OpenCVBrushStrokesNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE", {"tooltip": "Input image tensor from the Load Image node."}),
                "preset": (["Custom", "Expressionist", "Realist", "Abstract", "Impressionist"], {"tooltip": "Select a painterly preset. Choosing a preset overrides manual parameters."}),
                # Only used in Custom mode.
                "effect": (["Stylization", "Edge Preserving"], {"tooltip": "In Custom mode, choose the OpenCV effect to apply."}),
                "sigma_s": ("FLOAT", {"default": 60.0, "min": 0.0, "max": 200.0, "step": 1.0, "tooltip": "Spatial scale (sigma_s); used only in Custom mode."}),
                "sigma_r": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Range value (sigma_r); used only in Custom mode."}),
                "filter_flag": ("INT", {"default": 1, "min": 0, "max": 1, "step": 1, "tooltip": "For Edge Preserving: 0 for Normal, 1 for Recursive filter; used only in Custom mode."})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "Utility"
    
    def process_image(self, input_image: torch.Tensor, preset: str, effect: str, sigma_s: float, sigma_r: float, filter_flag: int):
        if preset != "Custom":
            if preset == "Expressionist":
                effect = "Stylization"
                sigma_s, sigma_r = 80.0, 0.50
            elif preset == "Realist":
                effect = "Edge Preserving"
                sigma_s, sigma_r, filter_flag = 120.0, 0.40, 1
            elif preset == "Abstract":
                effect = "Stylization"
                sigma_s, sigma_r = 150.0, 0.60
            elif preset == "Impressionist":
                effect = "Stylization"
                sigma_s, sigma_r = 90.0, 0.55
        img_np = input_image[0].cpu().numpy()
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
# 3. PILBrushStrokesNode (using native PIL enhancements)
###############################################
class PILBrushStrokesNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE", {"tooltip": "Input image tensor from the Load Image node."}),
                "preset": (["Custom", "Expressionist", "Realist", "Abstract", "Impressionist"], {"tooltip": "Select a painterly preset. Choosing a preset overrides manual parameters."}),
                # Only used in Custom mode.
                "effect": (["Color Painting", "Realistic", "Vibrance"], {"tooltip": "In Custom mode, choose the PIL effect to apply."}),
                "strength": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 1.0, "tooltip": "Controls effect intensity (e.g. saturation/brightness factor); used only in Custom mode."}),
                "radius": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 1.0, "tooltip": "Modulates posterization depth or contrast; used only in Custom mode."})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "Utility"
    
    def process_image(self, input_image: torch.Tensor, preset: str, effect: str, strength: float, radius: float):
        if preset != "Custom":
            if preset == "Expressionist":
                effect = "Color Painting"
                strength, radius = 25.0, 3.0
            elif preset == "Realist":
                effect = "Realistic"
                strength, radius = 15.0, 15.0
            elif preset == "Abstract":
                effect = "Vibrance"
                strength, radius = 30.0, 10.0
            elif preset == "Impressionist":
                effect = "Color Painting"
                strength, radius = 20.0, 5.0
        img_np = input_image[0].cpu().numpy()
        img_uint8 = (np.clip(img_np * 255.0, 0, 255)).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8).convert("RGB")
        if effect == "Color Painting":
            enhancer = ImageEnhance.Color(pil_img)
            factor = 1 + (strength / 10.0)
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

NODE_CLASS_MAPPINGS = {
    "PILBrushStrokesNode": PILBrushStrokesNode,
    "OpenCVBrushStrokesNode": OpenCVBrushStrokesNode,
    "WandBrushStrokesNode": WandBrushStrokesNode
}