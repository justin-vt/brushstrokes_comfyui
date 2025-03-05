from .nodes import PILBrushStrokesNode 
from .nodes import OpenCVBrushStrokesNode 
from .nodes import WandBrushStrokesNode 

NODE_CLASS_MAPPINGS = {
    "PILBrushStrokesNode": PILBrushStrokesNode,
    "OpenCVBrushStrokesNode": OpenCVBrushStrokesNode,
    "WandBrushStrokesNode": WandBrushStrokesNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PILBrushStrokesNode": "Brush Strokes (PIL bindings)",
    "OpenCVBrushStrokesNode": "Brush Strokes (OpenCV bindings)",
    "WandBrushStrokesNode": "Brush Strokes (ImageMagick bindings)"
}