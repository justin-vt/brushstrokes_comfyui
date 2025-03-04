from PIL import Image

class BrushStrokesNode:
    @classmethod
    def INPUT_TYPES(cls):
        # Note that the image type is provided as a tuple.
        return {"required": {"input_image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "pass_through"
    CATEGORY = "Utility"

    def pass_through(self, input_image):
        # Simply return the input image unchanged.
        return (input_image,)

NODE_CLASS_MAPPINGS = {
    "BrushStrokesNode": BrushStrokesNode
}
