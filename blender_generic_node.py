import os
import json
import subprocess

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "blender_tools_config.json")
WRAPPER_PATH = os.path.join(os.path.dirname(__file__), "wrapper.py")

# Load configuration
def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    else:
        return {"blender_path": ""}

class BlenderGenericNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "blend_file": ("STRING", {"multiline": False}),
                "input_model": ("STRING", {"multiline": False}),
                "config_file": ("STRING", {"multiline": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32-1, "step": 1, "widget": "seed"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run_blender_task"
    CATEGORY = "Blender"

    def run_blender_task(self, blend_file, input_model, config_file, seed):
        cfg = load_config()
        blender_exe = cfg.get("blender_path", "")
        if not blender_exe or not os.path.exists(blender_exe):
            return ("Blender path not set or invalid in blender_tools_config.json",)

        env = os.environ.copy()
        env["BLEND_FILE"] = blend_file
        env["INPUT_MODEL"] = input_model
        env["CONFIG_FILE"] = config_file

        cmd = [blender_exe, "--background", "--python", WRAPPER_PATH]

        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            return ("Blender run failed",)

        return ("Blender task complete",)

NODE_CLASS_MAPPINGS = {
    "BlenderGenericNode": BlenderGenericNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BlenderGenericNode": "Blender Generic Runner"
}