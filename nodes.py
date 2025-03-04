import io
from PIL import Image
from wand.image import Image as WandImage

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
            input_image (PIL.Image): The input image from the load image node.
            radius (float): The radius for the sketch effect.
            sigma (float): The sigma (blur intensity) for the sketch effect.
            angle (float): The angle for the sketch effect.
            
        Returns:
            A tuple with the modified image.
        """
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        input_image.save(img_byte_arr, format="PNG")
        img_bytes = img_byte_arr.getvalue()

        # Use Wand (ImageMagick) to apply the sketch effect
        with WandImage(blob=img_bytes) as wand_img:
            # The sketch effect simulates a brush stroke appearance.
            wand_img.sketch(radius=radius, sigma=sigma, angle=angle)
            result_blob = wand_img.make_blob("png")

        # Convert the result back to a PIL image
        result_image = Image.open(io.BytesIO(result_blob))
        return (result_image,)

NODE_CLASS_MAPPINGS = {
    "BrushStrokesNode": BrushStrokesNode
}
