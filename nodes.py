from PIL import Image
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
                "input_image": ("IMAGE",),
                # Library selection dropdown
                "library": (["ImageMagick", "OpenCV"],),
                # Brush style dropdown
                "style": (["Oil Paint", "Charcoal", "Sketch"],),
                # Numerical input for effect intensity as a textbox with a default value.
                "strength": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                # Numerical input for radius as a textbox with a default value.
                "radius": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 1.0})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "Utility"

    def process_image(self, input_image: torch.Tensor, library: str, style: str, strength: float, radius: float):
        # Convert input tensor (shape [1, H, W, 3]) with values in [0,1] to a uint8 numpy array.
        img_np = input_image[0].cpu().numpy()  # shape: [H, W, 3]
        img_uint8 = (np.clip(img_np * 255.0, 0, 255)).astype(np.uint8)

        if library == "ImageMagick":
            # Use wand (ImageMagick) for processing.
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

        else:
            processed_np = img_np

        output_tensor = torch.from_numpy(processed_np)[None, ...]
        return (output_tensor,)

NODE_CLASS_MAPPINGS = {
    "BrushStrokesNode": BrushStrokesNode
}
