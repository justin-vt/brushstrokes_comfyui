from PIL import Image
import numpy as np
import torch
import io
from wand.image import Image as WandImage

class BrushStrokesNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
                # Dropdown for brush style selection.
                "style": (["Oil Paint", "Charcoal", "Sketch"],),
                # Numerical parameter for effect intensity (used for sigma in some effects).
                "strength": "FLOAT",
                # Radius parameter for the effect.
                "radius": "FLOAT"
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "Utility"

    def process_image(self, input_image: torch.Tensor, style: str, strength: float, radius: float):
        # 1. Convert the torch tensor to a numpy array (removing the batch dimension)
        img_np = input_image[0].cpu().numpy()  # shape: [H, W, 3]
        # Convert normalized float values (0-1) to 0-255 uint8 values.
        img_uint8 = (np.clip(img_np * 255.0, 0, 255)).astype(np.uint8)

        # 2. Convert numpy array to a PIL Image.
        pil_img = Image.fromarray(img_uint8)

        # 3. Save the PIL image to a bytes buffer (PNG format).
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        buf.seek(0)

        # 4. Open image with wand and apply the chosen effect.
        with WandImage(blob=buf.getvalue()) as wand_img:
            if style == "Oil Paint":
                # For Oil Paint, use the radius parameter.
                wand_img.oil_paint(radius=radius)
            elif style == "Charcoal":
                # For Charcoal, use radius and use strength as sigma.
                wand_img.charcoal(radius=radius, sigma=strength)
            elif style == "Sketch":
                # For Sketch, use radius and use strength as sigma (with a fixed angle).
                wand_img.sketch(radius=radius, sigma=strength, angle=0)
            else:
                # If for some reason an unsupported style is selected, leave the image unchanged.
                pass

            # Retrieve the processed image as a PNG blob.
            processed_blob = wand_img.make_blob(format='png')

        # 5. Convert the processed blob back into a PIL Image.
        processed_pil = Image.open(io.BytesIO(processed_blob))
        processed_pil = processed_pil.convert("RGB")

        # 6. Convert the PIL image back to a numpy array with normalized float values.
        processed_np = np.array(processed_pil).astype(np.float32) / 255.0

        # 7. Convert the numpy array to a torch tensor and add the batch dimension.
        output_tensor = torch.from_numpy(processed_np)[None, ...]

        return (output_tensor,)

NODE_CLASS_MAPPINGS = {
    "BrushStrokesNode": BrushStrokesNode
}
