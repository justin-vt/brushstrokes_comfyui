import io
import numpy as np
from PIL import Image
from wand.image import Image as WandImage

def tensor_to_pil(tensor):
    """
    Converts a tensor (assumed to be in CHW format) into a PIL Image.
    If the tensor is a torch.Tensor, it is converted to a NumPy array.
    """
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu().numpy()
    tensor = np.array(tensor)
    # If tensor has 3 dimensions and channels are first, rearrange to HWC.
    if tensor.ndim == 3:
        if tensor.shape[0] in [1, 3, 4]:
            tensor = np.transpose(tensor, (1, 2, 0))
    # For floating point images, assume values in [0, 1]
    if np.issubdtype(tensor.dtype, np.floating):
        tensor = np.clip(tensor, 0, 1)
        tensor = (tensor * 255).astype(np.uint8)
    else:
        tensor = tensor.astype(np.uint8)
    return Image.fromarray(tensor)

def pil_to_tensor(pil_image):
    """
    Converts a PIL Image to a tensor in CHW format with values normalized between 0 and 1.
    """
    import torch
    np_image = np.array(pil_image)
    # If grayscale, add a channel dimension.
    if np_image.ndim == 2:
        np_image = np.expand_dims(np_image, axis=-1)
    # Convert from HWC to CHW.
    np_image = np.transpose(np_image, (2, 0, 1))
    if np_image.dtype == np.uint8:
        np_image = np_image.astype(np.float32) / 255.0
    tensor_image = torch.from_numpy(np_image)
    return tensor_image

class BrushStrokesNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"input_image": ("IMAGE",)},
            "optional": {
                "radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "sigma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "angle": ("FLOAT", {"default": 45.0, "min": 0.0, "max": 360.0, "step": 1.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_brush_strokes"
    CATEGORY = "Effects"

    def apply_brush_strokes(self, input_image, radius=0.0, sigma=1.0, angle=45.0):
        """
        Applies a brush stroke (sketch) effect using ImageMagick via Wand.
        
        Parameters:
            input_image (Tensor or PIL.Image): The input image from the load image node.
            radius (float): The radius for the sketch effect.
            sigma (float): The sigma (blur intensity) for the sketch effect.
            angle (float): The angle for the sketch effect.
            
        Returns:
            A tuple with the modified image in tensor format suitable for preview.
        """
        # Convert input_image to a PIL Image if it's a tensor.
        if not isinstance(input_image, Image.Image):
            pil_image = tensor_to_pil(input_image)
        else:
            pil_image = input_image

        # Convert the PIL image to a PNG byte stream.
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format="PNG")
        img_bytes = img_byte_arr.getvalue()

        # Use Wand (ImageMagick) to apply the sketch (brush stroke) effect.
        with WandImage(blob=img_bytes) as wand_img:
            wand_img.sketch(radius=radius, sigma=sigma, angle=angle)
            result_blob = wand_img.make_blob("png")

        # Convert the result back to a PIL image.
        result_image = Image.open(io.BytesIO(result_blob)).convert("RGB")
        
        # Convert the PIL image back to a tensor format suitable for preview.
        result_tensor = pil_to_tensor(result_image)
        return (result_tensor,)

NODE_CLASS_MAPPINGS = {
    "BrushStrokesNode": BrushStrokesNode
}
