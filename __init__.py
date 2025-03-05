from .nodes import PILBrushStrokesNode 
from .nodes import OpenCVBrushStrokesNode 
from .nodes import WandBrushStrokesNode 

NODE_CLASS_MAPPINGS = {
    "OpenCVBrushStrokesNode": OpenCVBrushStrokesNode\
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenCVBrushStrokesNode": "Brush Strokes (OpenCV bindings)"
}