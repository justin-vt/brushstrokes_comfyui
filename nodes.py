from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import torch
import io
from wand.image import Image as WandImage
import cv2

class WandBrushStrokesNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE", {"tooltip": "Input image tensor from the Load Image node."}),
                "style": (["Swirl", "Wave", "Pixelation"], {"tooltip": "Select the Wand effect: Swirl, Wave, or Pixelation."}),
                # Swirl: degree in [0,360]
                "degree": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 360.0, "step": 1.0, "tooltip": "For 'Swirl': the swirl degree."}),
                # Wave: amplitude and wavelength in pixels
                "amplitude": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 1.0, "tooltip": "For 'Wave': the wave amplitude in pixels."}),
                "wavelength": ("FLOAT", {"default": 50.0, "min": 1.0, "max": 200.0, "step": 1.0, "tooltip": "For 'Wave': the wavelength in pixels."}),
                # Pixelation: pixel size factor (the larger the value, the stronger the pixelation)
                "pixel_size": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 50.0, "step": 1.0, "tooltip": "For 'Pixelation': size factor for pixelation."})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "Utility"

    def process_image(self, input_image: torch.Tensor, style: str, degree: float, amplitude: float, wavelength: float, pixel_size: float):
        # Convert input tensor [1, H, W, 3] with values in [0,1] to uint8 numpy array.
        img_np = input_image[0].cpu().numpy()  # shape: [H, W, 3]
        img_uint8 = (np.clip(img_np * 255.0, 0, 255)).astype(np.uint8)
        # Convert to PIL Image.
        pil_img = Image.fromarray(img_uint8)
        # Save to a bytes buffer.
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        buf.seek(0)

        # Open image with Wand.
        with WandImage(blob=buf.getvalue()) as wand_img:
            if style == "Swirl":
                wand_img.swirl(degree=degree)
            elif style == "Wave":
                # wave effect using amplitude and wavelength
                wand_img.wave(amplitude=amplitude, wavelength=wavelength)
            elif style == "Pixelation":
                # Simulate pixelation by resizing down then up.
                orig_width, orig_height = wand_img.width, wand_img.height
                new_width = max(1, int(orig_width / pixel_size))
                new_height = max(1, int(orig_height / pixel_size))
                wand_img.resize(new_width, new_height)
                wand_img.resize(orig_width, orig_height)
            # Get the processed image as PNG.
            processed_blob = wand_img.make_blob(format='png')

        processed_pil = Image.open(io.BytesIO(processed_blob)).convert("RGB")
        processed_np = np.array(processed_pil).astype(np.float32) / 255.0
        output_tensor = torch.from_numpy(processed_np)[None, ...]
        return (output_tensor,)

from PIL import Image
import numpy as np
import torch
import cv2

class OpenCVBrushStrokesNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE", {"tooltip": "Input image tensor from the Load Image node."}),
                "style": (["Stylization", "Edge Preserving"], {"tooltip": "Select the OpenCV effect: Stylization or Edge Preserving Filter."}),
                # Parameters for stylization and edge preserving.
                "sigma_s": ("FLOAT", {"default": 60.0, "min": 0.0, "max": 200.0, "step": 1.0, "tooltip": "For both effects: spatial scale (sigma_s)."}),
                "sigma_r": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "For both effects: range value (sigma_r). For Edge Preserving, this controls smoothing."}),
                # Optional: flag for edge preserving filter (0 or 1) - defaulting to RECURS_FILTER.
                "filter_flag": ("INT", {"default": 1, "min": 0, "max": 1, "step": 1, "tooltip": "For 'Edge Preserving': 0 for Normal, 1 for Recursive filter."})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "Utility"

    def process_image(self, input_image: torch.Tensor, style: str, sigma_s: float, sigma_r: float, filter_flag: int):
        # Convert input tensor to uint8 numpy array.
        img_np = input_image[0].cpu().numpy()  # shape: [H, W, 3]
        img_uint8 = (np.clip(img_np * 255.0, 0, 255)).astype(np.uint8)
        # OpenCV works in BGR.
        cv_img = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

        if style == "Stylization":
            processed_cv = cv2.stylization(cv_img, sigma_s=sigma_s, sigma_r=sigma_r)
        elif style == "Edge Preserving":
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

class PILBrushStrokesNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE", {"tooltip": "Input image tensor from the Load Image node."}),
                "style": (["Color Painting", "Realistic", "Vibrance"], {"tooltip": "Select the PIL effect: Color Painting, Realistic, or Vibrance."}),
                # Parameters for PIL enhancements.
                "strength": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 1.0, "tooltip": "For enhancements: controls intensity (e.g. saturation or brightness factor)."}),
                "radius": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 1.0, "tooltip": "Used to modulate posterization depth or contrast adjustments."})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "Utility"

    def process_image(self, input_image: torch.Tensor, style: str, strength: float, radius: float):
        # Convert input tensor [1, H, W, 3] with values in [0,1] to a uint8 numpy array.
        img_np = input_image[0].cpu().numpy()  # shape: [H, W, 3]
        img_uint8 = (np.clip(img_np * 255.0, 0, 255)).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8).convert("RGB")

        if style == "Color Painting":
            # Increase saturation.
            enhancer = ImageEnhance.Color(pil_img)
            factor = 1 + (strength / 20.0)
            pil_img = enhancer.enhance(factor)
            # Apply posterize effect; map radius to posterize bits (1 to 8)
            bits = max(1, min(8, int(8 - (radius / 10.0))))
            pil_img = ImageOps.posterize(pil_img, bits)
        elif style == "Realistic":
            # Adjust brightness and contrast.
            enhancer = ImageEnhance.Brightness(pil_img)
            pil_img = enhancer.enhance(1 + (strength / 50.0))
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(1 + (radius / 20.0))
        elif style == "Vibrance":
            # Increase both saturation and contrast.
            enhancer = ImageEnhance.Color(pil_img)
            pil_img = enhancer.enhance(1 + (strength / 10.0))
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(1 + (radius / 20.0))
        else:
            # No effect selected.
            pass

        processed_np = np.array(pil_img).astype(np.float32) / 255.0
        output_tensor = torch.from_numpy(processed_np)[None, ...]
        return (output_tensor,)

NODE_CLASS_MAPPINGS = {
    "PILBrushStrokesNode": PILBrushStrokesNode,
    "OpenCVBrushStrokesNode": OpenCVBrushStrokesNode,
    "WandBrushStrokesNode": WandBrushStrokesNode
}