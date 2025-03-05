from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import torch
import io
from wand.image import Image as WandImage
import cv2

class BrushStrokesNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE", {"tooltip": "Input image tensor from the Load Image node."}),
                # Select the processing library.
                "library": (["ImageMagick", "OpenCV", "PIL"], {"tooltip": "Select the processing library: 'ImageMagick' (via wand) and 'OpenCV' support Oil Paint, Charcoal, and Sketch effects. 'PIL' uses native PIL enhancements to simulate Color Painting, Realistic, and Vibrance effects."}),
                # Brush style selection – valid choices depend on the selected library.
                "style": (["Oil Paint", "Charcoal", "Sketch", "Color Painting", "Realistic", "Vibrance"], {"tooltip": "Select the brush style effect. For ImageMagick/OpenCV: Oil Paint, Charcoal, Sketch. For PIL: Color Painting, Realistic, Vibrance."}),
                # Numerical parameter for effect intensity.
                "strength": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 1.0, "tooltip": "Effect intensity: For Charcoal/Sketch, this sets sigma (or used for other enhancements in PIL)."}),
                # Radius parameter controlling stroke scale.
                "radius": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 1.0, "tooltip": "Controls the spatial scale of the effect (i.e. brush stroke size, or used as posterize bits in PIL)."})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "Utility"

    def process_image(self, input_image: torch.Tensor, library: str, style: str, strength: float, radius: float):
        # Convert input tensor (shape [1, H, W, 3] with values in [0,1]) to a uint8 numpy array.
        img_np = input_image[0].cpu().numpy()  # shape: [H, W, 3]
        img_uint8 = (np.clip(img_np * 255.0, 0, 255)).astype(np.uint8)

        # --- Branch based on the selected library ---
        if library in ["ImageMagick", "OpenCV"]:
            # For ImageMagick and OpenCV, only the first three styles are supported.
            if style not in ["Oil Paint", "Charcoal", "Sketch"]:
                # If an unsupported style is chosen for these libraries, return the input unmodified.
                processed_np = img_np
            else:
                if library == "ImageMagick":
                    # Use Wand (ImageMagick) for processing.
                    pil_img = Image.fromarray(img_uint8)
                    buf = io.BytesIO()
                    pil_img.save(buf, format='PNG')
                    buf.seek(0)
                    with WandImage(blob=buf.getvalue()) as wand_img:
                        if style == "Oil Paint":
                            wand_img.oil_paint(radius=radius)
                        elif style == "Charcoal":
                            wand_img.charcoal(radius=radius, sigma=strength)
                        elif style == "Sketch":
                            wand_img.sketch(radius=radius, sigma=strength, angle=0)
                        processed_blob = wand_img.make_blob(format='png')
                    processed_pil = Image.open(io.BytesIO(processed_blob)).convert("RGB")
                    processed_np = np.array(processed_pil).astype(np.float32) / 255.0

                elif library == "OpenCV":
                    # Use OpenCV for processing.
                    cv_img = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
                    if style == "Oil Paint":
                        processed_cv = cv2.stylization(cv_img, sigma_s=radius, sigma_r=strength/100.0)
                    elif style == "Charcoal":
                        sketch_gray, sketch_color = cv2.pencilSketch(cv_img, sigma_s=radius, sigma_r=strength/100.0, shade_factor=0.05)
                        processed_cv = sketch_color
                    elif style == "Sketch":
                        sketch_gray, sketch_color = cv2.pencilSketch(cv_img, sigma_s=radius, sigma_r=strength/100.0, shade_factor=0.05)
                        processed_cv = cv2.cvtColor(sketch_gray, cv2.COLOR_GRAY2BGR)
                    processed_rgb = cv2.cvtColor(processed_cv, cv2.COLOR_BGR2RGB)
                    processed_np = processed_rgb.astype(np.float32) / 255.0

        elif library == "PIL":
            # Use PIL enhancements for realistic, color-rich effects.
            pil_img = Image.fromarray(img_uint8).convert("RGB")
            if style == "Color Painting":
                # Increase saturation and apply posterization.
                enhancer = ImageEnhance.Color(pil_img)
                factor = 1 + (strength / 20.0)  # Adjust factor based on strength.
                pil_img = enhancer.enhance(factor)
                bits = max(1, min(8, int(8 - radius)))  # Map radius to posterize bits.
                pil_img = ImageOps.posterize(pil_img, bits)
            elif style == "Realistic":
                # Adjust brightness and contrast.
                enhancer = ImageEnhance.Brightness(pil_img)
                pil_img = enhancer.enhance(1 + (strength / 50.0))
                enhancer = ImageEnhance.Contrast(pil_img)
                pil_img = enhancer.enhance(1 + (radius / 20.0))
            elif style == "Vibrance":
                # Increase saturation and contrast for a vibrant look.
                enhancer = ImageEnhance.Color(pil_img)
                pil_img = enhancer.enhance(1 + (strength / 10.0))
                enhancer = ImageEnhance.Contrast(pil_img)
                pil_img = enhancer.enhance(1 + (radius / 20.0))
            else:
                # If an unsupported PIL style is selected, leave the image unchanged.
                pass
            processed_np = np.array(pil_img).astype(np.float32) / 255.0

        else:
            # Fallback: return the input image unchanged.
            processed_np = img_np

        # Convert the processed numpy array back to a torch tensor with a batch dimension.
        output_tensor = torch.from_numpy(processed_np)[None, ...]
        return (output_tensor,)

NODE_CLASS_MAPPINGS = {
    "BrushStrokesNode": BrushStrokesNode
}
