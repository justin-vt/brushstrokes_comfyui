from PIL import Image
import numpy as np

class BrushStrokesNode:
    @classmethod
    def INPUT_TYPES(cls):
        # Accepts an image input; specifying as a tuple for compatibility.
        return {"required": {"input_image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "Utility"

    def process_image(self, input_image):
        # Convert input to a PIL image if it's not already one.
        if not isinstance(input_image, Image.Image):
            pil_image = Image.fromarray(input_image)
        else:
            pil_image = input_image

        # (Future manipulation can occur here using PIL's features.)

        # Convert the PIL image back to a numpy array for preview.
        output_image = np.array(pil_image)
        return (output_image,)

NODE_CLASS_MAPPINGS = {
    "BrushStrokesNode": BrushStrokesNode
}
