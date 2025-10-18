import os
import json
from server import PromptServer

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "blender_tools_config.json")



# Import node definitions
from .blender_generic_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]