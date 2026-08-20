In ```src/megapose/inference/utils.py```:

From
```python
def make_renderer(renderer_type: str) -> Panda3dBatchRenderer:
        logger.debug("renderer_kwargs", renderer_kwargs)
        if renderer_kwargs is None:
            renderer_kwargs_ = dict()
        else:
            renderer_kwargs_ = renderer_kwargs

        renderer_kwargs_.setdefault("split_objects", True)
        renderer_kwargs_.setdefault("preload_cache", False)
        renderer_kwargs_.setdefault("n_workers", 4)

        if renderer_type == "panda3d" or force_panda3d_renderer:
            renderer = Panda3dBatchRenderer(object_dataset=object_dataset, **renderer_kwargs_)
        else:
            raise ValueError(renderer_type)
        return renderer
```
To
```python
def make_renderer(renderer_type: str):
        from megapose.nvdiffrast.nvdiffrast_renderer import NvdiffrastSceneRenderer
        # Pass the raw object dataset so we can access the file paths
        return NvdiffrastSceneRenderer(object_dataset=object_dataset)
```

---

In ```src/megapose/models/pose_rigid.py```:

From
```python
def render_images_multiview(
        self,
        labels: List[str],
        TCV_O: torch.Tensor,
        KV: torch.Tensor,
        random_ambient_light: bool = False,
    ) -> torch.Tensor:
        """Render multiple images.

        Args:
            labels: list[str] with length bsz
            TCV_O: [bsz, n_views, 4, 4] pose of the cameras defining each view
            KV: [bsz, n_views, 4, 4] intrinsics of the associated cameras
            random_ambient_light: Whether to use randomize ambient light parameter.

        Returns
            renders: [bsz, n_views*n_channels, H, W]
        """

        labels_mv = []
        bsz = len(labels)
        n_views = TCV_O.shape[1]
        for n in range(bsz):
            for _ in range(n_views):
                labels_mv.append(labels[n])

        if random_ambient_light:
            light_datas = []
            for _ in range(len(labels_mv)):
                intensity = np.random.uniform(0.7, 1.0)
                lights = [
                    Panda3dLightData(
                        light_type="ambient",
                        color=(intensity, intensity, intensity, 1.0),
                    )
                ]
                light_datas.append(lights)
        else:
            if self.render_normals:
                ambient_light = Panda3dLightData(light_type="ambient", color=(1.0, 1.0, 1.0, 1.0))
                light_datas = [[ambient_light] for _ in range(len(labels_mv))]
            else:
                light_datas = [make_scene_lights() for _ in range(len(labels_mv))]

        assert isinstance(self.renderer, Panda3dBatchRenderer)

        render_mask = False

        render_data = self.renderer.render(
            labels=labels_mv,
            TCO=TCV_O.flatten(0, 1),
            K=KV.flatten(0, 1),
            render_mask=render_mask,
            resolution=self.render_size,
            render_normals=self.render_normals,
            render_depth=self.render_depth,
            light_datas=light_datas,
        )

        cat_list = []
        cat_list.append(render_data.rgbs)

        if self.render_normals:
            cat_list.append(render_data.normals)

        if self.render_depth:
            cat_list.append(render_data.depths)

        renders = torch.cat(cat_list, dim=1)
        n_channels = renders.shape[1]

        renders = renders.view(bsz, n_views, n_channels, *renders.shape[-2:]).flatten(1, 2)
        return renders  # [bsz, n_views*n_channels, H, W]
```
to
```python
def render_images_multiview(
        self,
        labels: List[str],
        TCV_O: torch.Tensor,
        KV: torch.Tensor,
        random_ambient_light: bool = False,
    ) -> torch.Tensor:
        """Single-threaded, zero-copy nvdiffrast replacement for Jetson/Edge Deployment.

        Args:
            labels: list[str] with length bsz
            TCV_O: [bsz, n_views, 4, 4] pose of the cameras defining each view
            KV: [bsz, n_views, 3, 3] intrinsics of the associated cameras
            random_ambient_light: Unused by nvdiffrast baseline.

        Returns:
            renders: [bsz, n_views*n_channels, H, W]
        """
        bsz = len(labels)
        n_views = TCV_O.shape[1]
        labels_mv = []
        for n in range(bsz):
            for _ in range(n_views):
                labels_mv.append(labels[n])

        # Flatten cameras and intrinsics to process view-by-view
        TCO_flat = TCV_O.flatten(0, 1).detach()
        K_flat = KV.flatten(0, 1).detach()
        
        # We must invert TCO to get TWC (Transform World to Camera) matrix for camera definition
        from megapose.lib3d.transform_ops import invert_transform_matrices
        from megapose.lib3d.transform import Transform
        from megapose.panda3d_renderer.types import Panda3dCameraData, Panda3dObjectData

        TOC_flat = invert_transform_matrices(TCO_flat)
        TWO = Transform((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0))

        # Lists to hold the separate native GPU channels
        list_rgbs = []
        list_normals = []
        list_depths = []

        # --- NATIVE CUDA RENDER LOOP ---
        for label_n, TOC_tensor, K_tensor in zip(labels_mv, TOC_flat, K_flat):
            
            # Wrap data into lightweight structures to maintain API compatibility
            # Convert tensors to CPU numpy arrays ONLY where the dataclass strictly expects it (like K or matrices)
            cam_data = Panda3dCameraData(
                TWC=Transform(TOC_tensor.cpu().numpy().astype(np.float32)), 
                K=K_tensor.cpu().numpy().astype(np.float32), 
                resolution=self.render_size
            )
            obj_data = Panda3dObjectData(label=label_n, TWO=TWO)

            # Fire the nvdiffrast scene renderer!
            renderings = self.renderer.render_scene(
                object_datas=[obj_data],
                camera_datas=[cam_data],
                light_datas=None,
                render_normals=self.render_normals,
                render_depth=self.render_depth,
                copy_arrays=False 
            )
            
            # 1. Grab native GPU RGB tensor [H, W, 3]
            list_rgbs.append(renderings[0].gpu_rgb)
            
            # 2. Grab native GPU Normals tensor [H, W, 3]
            if self.render_normals:
                list_normals.append(renderings[0].normals)
                
            # 3. Grab native GPU Depth tensor [H, W, 1]
            if self.render_depth:
                list_depths.append(renderings[0].depth)

        # --- STACK AND COMBINE CHANNELS IN VRAM ---
        cat_list = []
        
        # Stack RGBs, permute to (B, C, H, W) and normalize to [0, 1]
        rgbs = torch.stack(list_rgbs).float().permute(0, 3, 1, 2) / 255.0
        cat_list.append(rgbs)
        
        if self.render_normals:
            # Normals are already on CUDA, stack, permute, and normalize
            normals = torch.stack(list_normals).float().permute(0, 3, 1, 2) / 255.0
            cat_list.append(normals)
            
        if self.render_depth:
            # Depth maps stack directly
            depths = torch.stack(list_depths).float().permute(0, 3, 1, 2)
            cat_list.append(depths)

        # Reconstruct the 9-channel composite tensor along the channel axis (dim=1)
        renders = torch.cat(cat_list, dim=1)
        
        n_channels = renders.shape[1]
        renders = renders.view(bsz, n_views, n_channels, *renders.shape[-2:]).flatten(1, 2)
        
        return renders  # [bsz, n_views*n_channels, H, W]
```