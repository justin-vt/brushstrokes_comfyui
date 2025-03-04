from PIL import Image
import numpy as np
import torch

class BrushStrokesNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"input_image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "Utility"

    def process_image(self, input_image: torch.Tensor):
        # ComfyUI image tensor expected shape: [1, H, W, 3] with float32 values [0,1]
        # 1. Convert the torch tensor to a numpy array (remove the batch dimension)
        img_np = input_image[0].cpu().numpy()  # shape: [H, W, 3]

        # 2. Convert the normalized float values to 0-255 uint8 values
        img_uint8 = (np.clip(img_np * 255.0, 0, 255)).astype(np.uint8)

        # 3. Convert the numpy array to a PIL Image (now you can manipulate with PIL if needed)
        pil_image = Image.fromarray(img_uint8)

        # (Future PIL manipulation can be added here)

        # 4. Convert the PIL image back to a numpy array with float32 values in [0,1]
        np_img = np.array(pil_image).astype(np.float32) / 255.0

        # 5. Convert the numpy array to a torch tensor and add a batch dimension
        output_tensor = torch.from_numpy(np_img)[None, ...]

        return (output_tensor,)

NODE_CLASS_MAPPINGS = {
    "BrushStrokesNode": BrushStrokesNode
}
