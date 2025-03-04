from PIL import Image
import numpy as np
import torch
import io
from wand.image import Image as WandImage

class BrushStrokesNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"input_image": ("IMAGE",)}}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "Utility"

    def process_image(self, input_image: torch.Tensor):
        # 1. Convert torch tensor to numpy array (remove batch dimension)
        img_np = input_image[0].cpu().numpy()  # shape: [H, W, 3]
        # Convert normalized floats to 0-255 uint8
        img_uint8 = (np.clip(img_np * 255.0, 0, 255)).astype(np.uint8)

        # 2. Convert numpy array to PIL Image
        pil_img = Image.fromarray(img_uint8)

        # 3. Save PIL image to a bytes buffer
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        buf.seek(0)

        # 4. Use wand to open the image from the bytes blob
        with WandImage(blob=buf.getvalue()) as wand_img:
            # Apply the oil paint effect (adjust radius as needed)
            wand_img.oil_paint(radius=5)
            # Retrieve the processed image as a blob
            processed_blob = wand_img.make_blob(format='png')

        # 5. Convert the processed blob back into a PIL image
        processed_pil = Image.open(io.BytesIO(processed_blob))
        processed_pil = processed_pil.convert("RGB")

        # 6. Convert PIL image back to a numpy array with normalized float values
        processed_np = np.array(processed_pil).astype(np.float32) / 255.0

        # 7. Convert numpy array to a torch tensor with a batch dimension
        output_tensor = torch.from_numpy(processed_np)[None, ...]

        return (output_tensor,)

NODE_CLASS_MAPPINGS = {
    "BrushStrokesNode": BrushStrokesNode
}
