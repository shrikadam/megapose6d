import torch
import nvdiffrast.torch as dr
import numpy as np
import trimesh
from megapose.panda3d_renderer.types import CameraRenderingData

class NvdiffrastSceneRenderer:
    def __init__(self, object_dataset):
        self.glctx = dr.RasterizeCudaContext()
        
        # 1. Store paths AND the critical physical scaling factor
        self.mesh_paths = {obj.label: obj.mesh_path for obj in object_dataset.list_objects}
        self.scales = {obj.label: obj.scale for obj in object_dataset.list_objects}
        
        self.vertices = {}
        self.faces = {}
        self.vertex_normals = {}
        self.vertex_colors = {}

    def _get_opengl_projection_matrix(self, K, h, w, z_near=0.01, z_far=10.0):
        """Correctly maps OpenCV (Z-forward, Y-down) to NDC."""
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        P = torch.zeros((4, 4), dtype=torch.float32, device="cuda")
        P[0, 0] = 2.0 * fx / w
        P[0, 2] = 2.0 * cx / w - 1.0
        P[1, 1] = -2.0 * fy / h
        P[1, 2] = 1.0 - 2.0 * cy / h
        P[2, 2] = (z_far + z_near) / (z_far - z_near)
        P[2, 3] = -2.0 * z_far * z_near / (z_far - z_near)
        P[3, 2] = 1.0
        return P

    def render_scene(
        self, object_datas, camera_datas, light_datas=None, 
        render_normals=False, render_depth=False, copy_arrays=False
    ):
        cam = camera_datas[0]
        obj = object_datas[0]
        h, w = cam.resolution
        label = obj.label
        
        # --- THE SCALE FIX ---
        if label not in self.vertices:
            mesh = trimesh.load(self.mesh_paths[label], process=False)
            scale = self.scales[label]
            
            # Physically shrink the boat back to real-world meters
            scaled_vertices = mesh.vertices * scale
            self.vertices[label] = torch.tensor(scaled_vertices, dtype=torch.float32, device="cuda")
            self.faces[label] = torch.tensor(mesh.faces, dtype=torch.int32, device="cuda")
            
            if hasattr(mesh, 'vertex_normals') and len(mesh.vertex_normals) > 0:
                self.vertex_normals[label] = torch.tensor(mesh.vertex_normals, dtype=torch.float32, device="cuda")
            else:
                self.vertex_normals[label] = torch.zeros_like(self.vertices[label])
                
            self.vertex_colors[label] = torch.ones_like(self.vertices[label]) * 0.5 
        
        v_pos = self.vertices[label]
        faces = self.faces[label]
        v_colors = self.vertex_colors[label]
        
        # --- THE MATRIX INVERSE FIX ---
        # cam.TWC is actually TOC (Transform Camera in Object space)
        TOC = torch.tensor(cam.TWC.matrix).float().cuda() 
        TWO = torch.tensor(obj.TWO.matrix).float().cuda() 
        
        # Calculate TCO: TCW = TOC^-1. Then TCO = TCW @ TWO
        TCW = torch.inverse(TOC)
        TCO = TCW @ TWO
        
        K = torch.tensor(cam.K).float().cuda()
        P_proj = self._get_opengl_projection_matrix(K, h, w)
        
        # Apply Transforms
        v_pos_homo = torch.nn.functional.pad(v_pos, (0, 1), value=1.0)
        v_cam = v_pos_homo @ TCO.T
        v_clip = v_cam @ P_proj.T
        v_clip = v_clip.unsqueeze(0) 

        # --- BULLETPROOF BOUNDARY ---
        safe_v_clip = v_clip.to(torch.float32).contiguous()
        safe_faces = faces.to(torch.int32).contiguous()
        safe_v_colors = v_colors.unsqueeze(0).to(torch.float32).contiguous()

        rast_out, rast_out_db = dr.rasterize(self.glctx, safe_v_clip, safe_faces, resolution=[h, w])
        safe_rast_out = rast_out.to(torch.float32).contiguous()

        # Silhouette Mask Extraction
        mask = (safe_rast_out[0, :, :, 3] > 0).unsqueeze(-1)

        # RGB Interpolation & Cleanup
        rgb, _ = dr.interpolate(safe_v_colors, safe_rast_out, safe_faces)
        rgb_out = (rgb[0] * 255).clamp(0, 255).to(torch.uint8)
        rgb_out = torch.where(mask, rgb_out, torch.zeros_like(rgb_out)) 
        
        rendering = CameraRenderingData(rgb=np.zeros((1,1,3), dtype=np.uint8))
        rendering.gpu_rgb = rgb_out.flip(0)

        # Normal Interpolation
        if render_normals:
            v_norms = self.vertex_normals[label]
            v_norms_rotated = v_norms @ TCO[:3, :3].T 
            safe_v_norms = v_norms_rotated.unsqueeze(0).to(torch.float32).contiguous()
            
            normals, _ = dr.interpolate(safe_v_norms, safe_rast_out, safe_faces)
            normals = torch.nn.functional.normalize(normals, dim=-1)
            
            normals_mapped = (normals[0] + 1.0) * 0.5
            normals_out = (normals_mapped * 255).clamp(0, 255).to(torch.uint8)
            normals_out = torch.where(mask, normals_out, torch.zeros_like(normals_out))
            
            rendering.normals = normals_out.flip(0)
        
        # Depth Interpolation
        if render_depth:
            depth = v_cam[:, 2].unsqueeze(0).unsqueeze(-1)
            safe_depth = depth.to(torch.float32).contiguous()
            
            depth_pixels, _ = dr.interpolate(safe_depth, safe_rast_out, safe_faces)
            depth_out = depth_pixels[0]
            depth_out = torch.where(mask, depth_out, torch.zeros_like(depth_out))
            
            rendering.depth = depth_out.flip(0)

        return [rendering]