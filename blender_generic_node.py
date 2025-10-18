import os
import json
import subprocess
import tempfile

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

    RETURN_TYPES = ("STRING", "DICT")
    RETURN_NAMES = ("fbx_path", "output_json")
    FUNCTION = "run_blender_task"
    CATEGORY = "Blender"

    def run_blender_task(self, blend_file, input_model, config_file, seed):
        cfg = load_config()
        blender_exe = cfg.get("blender_path", "")
        if not blender_exe or not os.path.exists(blender_exe):
            return ("Blender path not set or invalid in blender_tools_config.json",)

        # Create temporary output file
        tmp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tmp_output_path = tmp_output.name
        tmp_output.close()


        env = os.environ.copy()
        env["BLEND_FILE"] = blend_file
        env["INPUT_MODEL"] = input_model
        env["CONFIG_FILE"] = config_file
        env["OUTPUT_JSON"] = tmp_output_path 

        cmd = [blender_exe, "--background", "--python", WRAPPER_PATH]

        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        print(result.stdout)
        output_data = {}
        fbx_path = "No FBX Exported or not set"

        if os.path.exists(tmp_output_path):
            try:
                with open(tmp_output_path, "r") as f:
                    output_data = json.load(f)
                fbx_path = output_data.get("fbx_export", "")
            except Exception as e:
                print(f"Failed to parse output JSON: {e}")

        return (fbx_path, output_data)

NODE_CLASS_MAPPINGS = {
    "BlenderGenericNode": BlenderGenericNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BlenderGenericNode": "Blender Generic Runner"
}