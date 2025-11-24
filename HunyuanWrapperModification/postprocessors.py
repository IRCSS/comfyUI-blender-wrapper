# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import tempfile
import os
from typing import Union
import torch
import numpy as np
import pymeshlab
import trimesh
import time       # <-- add
import uuid       # <-- add

from .models.autoencoders import Latent2MeshOutput

import folder_paths

def _safe_unlink(path: str, tries: int = 10, delay: float = 0.1):
    """
    Best-effort removal of a temp file, with retries to avoid Windows
    'file in use' race conditions. Swallows PermissionError after retries.
    """
    for _ in range(tries):
        try:
            if os.path.exists(path):
                os.remove(path)
            return
        except PermissionError:
            time.sleep(delay)
        except OSError:
            # Other OS errors – stop trying
            return
    # Last attempt, ignore any error
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def reduce_face(mesh: pymeshlab.MeshSet, max_facenum: int = 200000):
    if max_facenum > mesh.current_mesh().face_number():
        return mesh

    mesh.apply_filter(
        "meshing_decimation_quadric_edge_collapse",
        targetfacenum=max_facenum,
        qualitythr=1.0,
        preserveboundary=True,
        boundaryweight=3,
        preservenormal=True,
        preservetopology=True,
        autoclean=True
    )
    return mesh


def remove_floater(mesh: pymeshlab.MeshSet):
    mesh.apply_filter("compute_selection_by_small_disconnected_components_per_face",
                      nbfaceratio=0.005)
    mesh.apply_filter("compute_selection_transfer_face_to_vertex", inclusive=False)
    mesh.apply_filter("meshing_remove_selected_vertices_and_faces")
    return mesh


def pymeshlab2trimesh(mesh: pymeshlab.MeshSet):
    """
    Convert a pymeshlab.MeshSet to a trimesh.Trimesh *in memory*,
    no temp files.
    """
    m = mesh.current_mesh()
    vertices = m.vertex_matrix()
    faces = m.face_matrix()

    # Construct a trimesh mesh directly from the arrays
    return trimesh.Trimesh(vertices=vertices, faces=faces)



def trimesh2pymeshlab(mesh: trimesh.Trimesh):
    """
    Convert a trimesh.Trimesh (or Scene) to a pymeshlab.MeshSet *in memory*,
    no temp files, no disk IO.
    """
    # Handle scene with multiple geometries
    if isinstance(mesh, trimesh.scene.Scene):
        temp_mesh = None
        for idx, obj in enumerate(mesh.geometry.values()):
            if idx == 0:
                temp_mesh = obj
            else:
                temp_mesh = temp_mesh + obj
        mesh = temp_mesh

    # trimesh stores vertices/faces as numpy arrays already
    vertices = mesh.vertices
    faces = mesh.faces

    ms = pymeshlab.MeshSet()
    pm = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)
    ms.add_mesh(pm, "converted_trimesh")

    return ms




def export_mesh(input, output):
    if isinstance(input, pymeshlab.MeshSet):
        mesh = output
    elif isinstance(input, Latent2MeshOutput):
        output = Latent2MeshOutput()
        output.mesh_v = output.current_mesh().vertex_matrix()
        output.mesh_f = output.current_mesh().face_matrix()
        mesh = output
    else:
        mesh = pymeshlab2trimesh(output)
    return mesh


def import_mesh(mesh: Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput, str]) -> pymeshlab.MeshSet:
    if isinstance(mesh, str):
        mesh = load_mesh(mesh)
    elif isinstance(mesh, Latent2MeshOutput):
        mesh = pymeshlab.MeshSet()
        mesh_pymeshlab = pymeshlab.Mesh(vertex_matrix=mesh.mesh_v, face_matrix=mesh.mesh_f)
        mesh.add_mesh(mesh_pymeshlab, "converted_mesh")

    if isinstance(mesh, (trimesh.Trimesh, trimesh.scene.Scene)):
        mesh = trimesh2pymeshlab(mesh)

    return mesh

def bpt_remesh(self, mesh: trimesh.Trimesh, verbose: bool = False, with_normal: bool = True, temperature: float = 0.5, batch_size: int = 1, pc_num: int = 4096, seed: int = 1234, samples: int = 50000):
        from .bpt.model import data_utils
        from .bpt.model.model import MeshTransformer
        from .bpt.model.serializaiton import BPT_deserialize
        from .bpt.utils import sample_pc, joint_filter

        pc_normal = sample_pc(mesh, pc_num=pc_num, with_normal=with_normal, seed=seed, samples=samples)

        pc_normal = pc_normal[None, :, :] if len(pc_normal.shape) == 2 else pc_normal

        from torch.serialization import add_safe_globals
        from deepspeed.runtime.fp16.loss_scaler import LossScaler
        from deepspeed.runtime.zero.config import ZeroStageEnum
        from deepspeed.utils.tensor_fragment import fragment_address

        add_safe_globals([LossScaler, fragment_address, ZeroStageEnum])

        model = MeshTransformer()

        comfyui_dir = os.path.dirname(os.path.abspath(__file__)) 
        model_path = os.path.join(comfyui_dir, 'bpt/bpt-8-16-500m.pt')
        print(model_path)
        model.load(model_path)
        model = model.eval().cuda().half()

        import torch
        pc_tensor = torch.from_numpy(pc_normal).cuda().half()
        if len(pc_tensor.shape) == 2:
            pc_tensor = pc_tensor.unsqueeze(0)

        codes = model.generate(
            pc=pc_tensor,
            filter_logits_fn=joint_filter,
            filter_kwargs=dict(k=50, p=0.95),
            return_codes=True,
            temperature=temperature,
            batch_size=batch_size,
        )
        
        coords = []
        try:
            for i in range(len(codes)):
                code = codes[i]
                code = code[code != model.pad_id].cpu().numpy()
                vertices = BPT_deserialize(
                    code,
                    block_size=model.block_size,
                    offset_size=model.offset_size,
                    use_special_block=model.use_special_block,
                )
                coords.append(vertices)
        except:
            coords.append(np.zeros(3, 3))

        vertices = coords[i]
        faces = torch.arange(1, len(vertices) + 1).view(-1, 3)

        # Move to CPU
        faces = faces.cpu().numpy()

        del model

        return data_utils.to_mesh(vertices, faces, transpose=False, post_process=True)

        
class BptMesh:
    def __call__(
        self,
        mesh: Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput, str],
        temperature: float = 0.5,
        batch_size: int = 1,
        with_normal: bool = True,
        verbose: bool = False,
        pc_num: int = 4096,
        seed: int = 1234,
        samples: int = 50000
    ) -> Union[pymeshlab.MeshSet, trimesh.Trimesh]:
        mesh = bpt_remesh(self, mesh=mesh, temperature=temperature, batch_size=batch_size, with_normal=with_normal, pc_num=pc_num, seed=seed, samples=samples)
        return mesh

class FaceReducer:
    def __call__(
        self,
        mesh: Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput, str],
        max_facenum: int = 40000
    ) -> Union[pymeshlab.MeshSet, trimesh.Trimesh]:
        ms = import_mesh(mesh)
        ms = reduce_face(ms, max_facenum=max_facenum)
        mesh = export_mesh(mesh, ms)
        return mesh


class FloaterRemover:
    def __call__(
        self,
        mesh: Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput, str],
    ) -> Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput]:
        ms = import_mesh(mesh)
        ms = remove_floater(ms)
        mesh = export_mesh(mesh, ms)
        return mesh


class DegenerateFaceRemover:
    def __call__(
        self,
        mesh: Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput, str],
    ) -> Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput]:
        ms = import_mesh(mesh)

        # Create temp file with explicit closing
        temp_file = tempfile.NamedTemporaryFile(suffix='.ply', delete=False)
        temp_file_path = temp_file.name
        temp_file.close()

        try:
            ms.save_current_mesh(temp_file_path)
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(temp_file_path)
        finally:
            _safe_unlink(temp_file_path)

        mesh = export_mesh(mesh, ms)
        return mesh


def mesh_normalize(mesh):
    """
    Normalize mesh vertices to sphere
    """
    scale_factor = 1.2
    vtx_pos = np.asarray(mesh.vertices)
    max_bb = (vtx_pos - 0).max(0)[0]
    min_bb = (vtx_pos - 0).min(0)[0]

    center = (max_bb + min_bb) / 2

    scale = torch.norm(torch.tensor(vtx_pos - center, dtype=torch.float32), dim=1).max() * 2.0

    vtx_pos = (vtx_pos - center) * (scale_factor / float(scale))
    mesh.vertices = vtx_pos

    return mesh


class MeshSimplifier:
    def __init__(self, executable: str = None):
        if executable is None:
            CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
            executable = os.path.join(CURRENT_DIR, "mesh_simplifier.bin")
        self.executable = executable

    def __call__(
        self,
        mesh: Union[trimesh.Trimesh],
    ) -> Union[trimesh.Trimesh]:
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as temp_input:
            with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as temp_output:
                mesh.export(temp_input.name)
                os.system(f'{self.executable} {temp_input.name} {temp_output.name}')
                ms = trimesh.load(temp_output.name, process=False)
                if isinstance(ms, trimesh.Scene):
                    combined_mesh = trimesh.Trimesh()
                    for geom in ms.geometry.values():
                        combined_mesh = trimesh.util.concatenate([combined_mesh, geom])
                    ms = combined_mesh
                ms = mesh_normalize(ms)
                return ms
