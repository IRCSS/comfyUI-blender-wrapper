import bpy
import os
import json
import traceback

blend_file = os.getenv("BLEND_FILE")
input_model = os.getenv("INPUT_MODEL")
config_path = os.getenv("CONFIG_FILE")

print("=== [Blender Wrapper] Starting ===")

# -------------------------------------------------------------
# Load config file if provided
# -------------------------------------------------------------
config = {}

if config_path and os.path.exists(config_path):
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"Loaded config from {config_path}:")
        print(json.dumps(config, indent=2))
    except Exception as e:
        print(f"Failed to read config file: {e}")
else:
    print("No config file provided or path invalid.")

# Optionally store globally in Blender
bpy.app.driver_namespace["config"] = config


# -------------------------------------------------------------
# Import model if provided
# -------------------------------------------------------------
input_objects = []

if input_model and os.path.exists(input_model):
    print(f"Importing model: {input_model}")
    ext = os.path.splitext(input_model)[1].lower()

    input_collection = bpy.data.collections.get("INPUT")
    if not input_collection:
        input_collection = bpy.data.collections.new("INPUT")
        bpy.context.scene.collection.children.link(input_collection)

    # Track objects before import
    before_import = set(bpy.data.objects)

    # Perform import
    if ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=input_model)
    elif ext == ".obj":
        bpy.ops.import_scene.obj(filepath=input_model)
    elif ext in [".glb", ".gltf"]:
        bpy.ops.import_scene.gltf(filepath=input_model)
    else:
        print(f"Unknown input format: {ext}")

    # Detect new objects
    after_import = set(bpy.data.objects)
    imported_objects = list(after_import - before_import)

    # Move them to INPUT collection
    for obj in imported_objects:
        if obj.name not in input_collection.objects:
            input_collection.objects.link(obj)
        for coll in obj.users_collection:
            if coll != input_collection:
                try:
                    coll.objects.unlink(obj)
                except:
                    pass

    # Collect meshes only
    input_objects = [obj for obj in imported_objects if obj.type == "MESH"]

else:
    print("No input model specified or path invalid.")

# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------

def safe_exec(text_block, name="<unknown>"):
    """Run a Blender text block safely with error capture."""
    try:
        print(f"Running script: {name}")
        exec(
            text_block.as_string(),
            {
                "__name__": "__main__",
                "blend_file": blend_file,
                "input_model": input_model,
                "config_path": config_path,
                "config": config,  # ← Inject config here
                "input_objects": input_objects,
            },
        )
        print(f"Finished script: {name}")
    except Exception as e:
        print(f"Error in script {name}: {e}")
        traceback.print_exc()

def append_all_from_blend(path):
    """Append all data blocks from a .blend file into the current scene."""
    if not os.path.exists(path):
        print(f"Warning Blend file not found: {path}")
        return

    print(f"Appending all from: {path}")
    with bpy.data.libraries.load(path, link=False) as (data_from, data_to):
        # Collect all datablock categories present in the .blend
        for attr in dir(data_from):
            if not attr.startswith("_"):
                try:
                    values = getattr(data_from, attr)
                    if isinstance(values, list) and values:
                        setattr(data_to, attr, values)
                        print(f"  → Appended {len(values)} {attr}")
                except Exception:
                    pass

    # Link objects to the current scene
    for obj in getattr(data_to, "objects", []):
        try:
            bpy.context.scene.collection.objects.link(obj)
        except Exception:
            pass

def run_text_scripts():
    """Run Main or all text blocks."""
    if "Main" in bpy.data.texts:
        print("Running Main script")
        safe_exec(bpy.data.texts["Main"], "Main")
    else:
        print("No Main found — running all text blocks")
        for txt in bpy.data.texts:
            safe_exec(txt, txt.name)



# -------------------------------------------------------------
# Load referenced blend file and run its scripts
# -------------------------------------------------------------
if blend_file and os.path.exists(blend_file):
    append_all_from_blend(blend_file)
    run_text_scripts()
else:
    print("No blend file specified or path invalid.")

print("=== [Blender Wrapper] Finished ===")