import torch
import nvdiffrast.torch as dr
import numpy as np
import trimesh
from megapose.panda3d_renderer.types import CameraRenderingData

class NvdiffrastSceneRenderer:
    def __init__(self, object_dataset):
        # 1. Initialize the CUDA Rasterization Context
        self.glctx = dr.RasterizeCudaContext()
        
        # 2. Map object labels to their physical hard drive paths
        self.mesh_paths = {obj.label: obj.mesh_path for obj in object_dataset.list_objects}
        
        # 3. Empty dictionaries for Lazy Loading
        self.vertices = {}
        self.faces = {}
        self.vertex_normals = {}
        self.vertex_colors = {}

    def _get_opengl_projection_matrix(self, K, h, w, z_near=0.01, z_far=10.0):
        # ... (keep your existing projection matrix code here) ...
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        P = torch.zeros((4, 4), dtype=torch.float32, device="cuda")
        P[0, 0] = 2.0 * fx / w
        P[1, 1] = 2.0 * fy / h
        P[0, 2] = 1.0 - 2.0 * cx / w
        P[1, 2] = 2.0 * cy / h - 1.0
        P[2, 2] = -(z_far + z_near) / (z_far - z_near)
        P[2, 3] = -2.0 * z_far * z_near / (z_far - z_near)
        P[3, 2] = -1.0
        return P

    def render_scene(
        self, object_datas, camera_datas, light_datas=None, 
        render_normals=False, render_depth=False, copy_arrays=False
    ):
        cam = camera_datas[0]
        obj = object_datas[0]
        h, w = cam.resolution
        label = obj.label
        
        # --- LAZY LOADING CACHE (WITH TRIMESH) ---
        if label not in self.vertices:
            # Load the raw mesh file from the hard drive
            mesh = trimesh.load(self.mesh_paths[label], process=False)
            
            # Extract true vertices and faces straight to VRAM
            self.vertices[label] = torch.tensor(mesh.vertices, dtype=torch.float32, device="cuda")
            self.faces[label] = torch.tensor(mesh.faces, dtype=torch.int32, device="cuda")
            
            # Extract or compute normals
            if hasattr(mesh, 'vertex_normals') and len(mesh.vertex_normals) > 0:
                self.vertex_normals[label] = torch.tensor(mesh.vertex_normals, dtype=torch.float32, device="cuda")
            else:
                self.vertex_normals[label] = torch.zeros_like(self.vertices[label])
                
            self.vertex_colors[label] = torch.ones_like(self.vertices[label]) * 0.5 
        
        # 1. Fetch native PyTorch mesh data
        v_pos = self.vertices[label]
        faces = self.faces[label]
        v_colors = self.vertex_colors[label]
        
        # ... (Keep the rest of your render_scene math and dr.rasterize code exactly the same below this line!) ...
        
        # 2. Math: Object Space -> Camera Space -> Clip Space
        TCO = torch.tensor(cam.TWC.matrix).float().cuda() 
        K = torch.tensor(cam.K).float().cuda()
        P_proj = self._get_opengl_projection_matrix(K, h, w)
        
        v_pos_homo = torch.nn.functional.pad(v_pos, (0, 1), value=1.0)
        v_cam = v_pos_homo @ TCO.T
        v_clip = v_cam @ P_proj.T
        v_clip = v_clip.unsqueeze(0) 

        # Force strict FP32/INT32 and contiguous memory exactly at the hardware handoff
        safe_v_clip = v_clip.to(torch.float32).contiguous()
        safe_faces = faces.to(torch.int32).contiguous()
        safe_v_colors = v_colors.unsqueeze(0).to(torch.float32).contiguous()

        # 3. Hardware Rasterization
        rast_out, rast_out_db = dr.rasterize(self.glctx, safe_v_clip, safe_faces, resolution=[h, w])
        safe_rast_out = rast_out.to(torch.float32).contiguous()

        # --- THE PHANTOM WALL FIX: EXTRACT SILHOUETTE MASK ---
        # The 4th channel of rast_out contains the triangle ID. >0 means object, 0 means background.
        # Shape is [1, H, W, 4], we need [H, W, 1] for broadcasting
        mask = (safe_rast_out[0, :, :, 3] > 0).unsqueeze(-1)

        # 4. Interpolate RGB
        rgb, _ = dr.interpolate(safe_v_colors, safe_rast_out, safe_faces)
        rgb_out = (rgb[0] * 255).clamp(0, 255).to(torch.uint8)
        rgb_out = torch.where(mask, rgb_out, torch.zeros_like(rgb_out)) # Clear background
        
        # --- THE OPENGL FLIP ---
        rendering = CameraRenderingData(rgb=np.zeros((1,1,3), dtype=np.uint8))
        rendering.gpu_rgb = rgb_out.flip(0)

        # 4b. Interpolate Normals
        if render_normals:
            v_norms = self.vertex_normals[label]
            v_norms_rotated = v_norms @ TCO[:3, :3].T 
            safe_v_norms = v_norms_rotated.unsqueeze(0).to(torch.float32).contiguous()
            
            normals, _ = dr.interpolate(safe_v_norms, safe_rast_out, safe_faces)
            normals = torch.nn.functional.normalize(normals, dim=-1)
            
            normals_mapped = (normals[0] + 1.0) * 0.5
            normals_out = (normals_mapped * 255).clamp(0, 255).to(torch.uint8)
            
            # CRITICAL: Remove phantom background normals!
            normals_out = torch.where(mask, normals_out, torch.zeros_like(normals_out))
            rendering.normals = normals_out.flip(0)
        
        # 4c. Interpolate Depth
        if render_depth:
            depth = v_cam[:, 2].unsqueeze(0).unsqueeze(-1)
            safe_depth = depth.to(torch.float32).contiguous()
            
            depth_pixels, _ = dr.interpolate(safe_depth, safe_rast_out, safe_faces)
            depth_out = depth_pixels[0]
            
            depth_out = torch.where(mask, depth_out, torch.zeros_like(depth_out))
            rendering.depth = depth_out.flip(0)

        return [rendering]