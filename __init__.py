from . import nodes

NODE_CLASS_MAPPINGS = {
    "BrushStrokesNode": nodes.BrushStrokesNode,
}
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
