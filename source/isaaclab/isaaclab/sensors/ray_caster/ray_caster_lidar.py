# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import re
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
import omni.physics.tensors.impl.api as physx
import warp as wp
from isaacsim.core.prims import XFormPrim
from isaacsim.core.simulation_manager import SimulationManager
from pxr import UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils.math import convert_quat, quat_apply, quat_apply_yaw
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh

from ..sensor_base import SensorBase
from .ray_caster_box_data import RayCasterBoxData

if TYPE_CHECKING:
    from .ray_caster_lidar_cfg import RayCasterLidarCfg

from .data_collector import SimulationDataSaver

class RayCasterLidar(SensorBase):
    """A ray-casting sensor.

    The ray-caster uses a set of rays to detect collisions with meshes in the scene. The rays are
    defined in the sensor's local coordinate frame. The sensor can be configured to ray-cast against
    a set of meshes with a given ray pattern.

    The meshes are parsed from the list of primitive paths provided in the configuration. These are then
    converted to warp meshes and stored in the `warp_meshes` list. The ray-caster then ray-casts against
    these warp meshes using the ray pattern provided in the configuration.

    .. note::
        Currently, only static meshes are supported. Extending the warp mesh to support dynamic meshes
        is a work in progress.
    """

    cfg: RayCasterLidarCfg
    """The configuration parameters."""

    def __init__(self, cfg: RayCasterLidarCfg):
        """Initializes the ray-caster object.

        Args:
            cfg: The configuration parameters.
        """
        # check if sensor path is valid
        # note: currently we do not handle environment indices if there is a regex pattern in the leaf
        #   For example, if the prim path is "/World/Sensor_[1,2]".
        sensor_path = cfg.prim_path.split("/")[-1]
        sensor_path_is_regex = re.match(r"^[a-zA-Z0-9/_]+$", sensor_path) is None
        if sensor_path_is_regex:
            raise RuntimeError(
                f"Invalid prim path for the ray-caster sensor: {self.cfg.prim_path}."
                "\n\tHint: Please ensure that the prim path does not contain any regex patterns in the leaf."
            )
        # Initialize base class
        super().__init__(cfg)
        # Create empty variables for storing output data
        self._data = RayCasterBoxData()
        # the warp meshes used for raycasting.
        self.meshes: dict[str, wp.Mesh] = {}
        
        # check the data box clip params setting
        if self.cfg.data_box_clip is not None:
            for i, val in enumerate(self.cfg.data_box_clip):
                assert isinstance(val, (int, float)) and val > 0, f"Element {i} of data_box_clip must be a positive number, got {val}"
        
        # check the data collection mode flag
        if cfg.data_collection:
            assert cfg.data_save_path is not None, "Must set data_save_path while data_collection is True !!!"
            self.data_saver = SimulationDataSaver(cfg.data_save_path, 'pcd', 'partial')

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Ray-caster @ '{self.cfg.prim_path}': \n"
            f"\tview type            : {self._view.__class__}\n"
            f"\tupdate period (s)    : {self.cfg.update_period}\n"
            f"\tnumber of meshes     : {len(self.meshes)}\n"
            f"\tnumber of sensors    : {self._view.count}\n"
            f"\tnumber of rays/sensor: {self.num_rays}\n"
            f"\ttotal number of rays : {self.num_rays * self._view.count}"
        )

    """
    Properties
    """

    @property
    def num_instances(self) -> int:
        return self._view.count

    @property
    def data(self) -> RayCasterData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        # reset the timers and counters
        super().reset(env_ids)
        # resolve None
        if env_ids is None:
            env_ids = slice(None)
            num_envs_ids = self._view.count
        else:
            num_envs_ids = len(env_ids)
        # resample the drift
        r = torch.empty(num_envs_ids, 3, device=self.device)
        self.drift[env_ids] = r.uniform_(*self.cfg.drift_range)
        # resample the height drift
        range_list = [self.cfg.ray_cast_drift_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
        ranges = torch.tensor(range_list, device=self.device)
        self.ray_cast_drift[env_ids] = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (num_envs_ids, 3), device=self.device
        )
        # data collection mode
        if self.cfg.data_collection:
            self.data_saver.update_period()
        
    """
    Implementation.
    """

    def _initialize_impl(self):
        super()._initialize_impl()
        # obtain global simulation view
        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        # check if the prim at path is an articulated or rigid prim
        # we do this since for physics-based view classes we can access their data directly
        # otherwise we need to use the xform view class which is slower
        found_supported_prim_class = False
        prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if prim is None:
            raise RuntimeError(f"Failed to find a prim at path expression: {self.cfg.prim_path}")
        # create view based on the type of prim
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            self._view = self._physics_sim_view.create_articulation_view(self.cfg.prim_path.replace(".*", "*"))
            found_supported_prim_class = True
        elif prim.HasAPI(UsdPhysics.RigidBodyAPI):
            self._view = self._physics_sim_view.create_rigid_body_view(self.cfg.prim_path.replace(".*", "*"))
            found_supported_prim_class = True
        else:
            self._view = XFormPrim(self.cfg.prim_path, reset_xform_properties=False)
            found_supported_prim_class = True
            omni.log.warn(f"The prim at path {prim.GetPath().pathString} is not a physics prim! Using XFormPrim.")
        # check if prim view class is found
        if not found_supported_prim_class:
            raise RuntimeError(f"Failed to find a valid prim view class for the prim paths: {self.cfg.prim_path}")

        # load the meshes by parsing the stage
        self._initialize_warp_meshes()
        # initialize the ray start and directions
        self._initialize_rays_impl()

    def _initialize_warp_meshes(self):
        # check number of mesh prims provided
        if len(self.cfg.mesh_prim_paths) != 1:
            raise NotImplementedError(
                f"RayCaster currently only supports one mesh prim. Received: {len(self.cfg.mesh_prim_paths)}"
            )

        # read prims to ray-cast
        for mesh_prim_path in self.cfg.mesh_prim_paths:
            # check if the prim is a plane - handle PhysX plane as a special case
            # if a plane exists then we need to create an infinite mesh that is a plane
            mesh_prim = sim_utils.get_first_matching_child_prim(
                mesh_prim_path, lambda prim: prim.GetTypeName() == "Plane"
            )
            # if we did not find a plane then we need to read the mesh
            if mesh_prim is None:
                # obtain the mesh prim
                mesh_prim = sim_utils.get_first_matching_child_prim(
                    mesh_prim_path, lambda prim: prim.GetTypeName() == "Mesh"
                )
                # check if valid
                if mesh_prim is None or not mesh_prim.IsValid():
                    raise RuntimeError(f"Invalid mesh prim path: {mesh_prim_path}")
                # cast into UsdGeomMesh
                mesh_prim = UsdGeom.Mesh(mesh_prim)
                # read the vertices and faces
                points = np.asarray(mesh_prim.GetPointsAttr().Get())
                transform_matrix = np.array(omni.usd.get_world_transform_matrix(mesh_prim)).T
                points = np.matmul(points, transform_matrix[:3, :3].T)
                points += transform_matrix[:3, 3]
                indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get())
                wp_mesh = convert_to_warp_mesh(points, indices, device=self.device)
                # print info
                omni.log.info(
                    f"Read mesh prim: {mesh_prim.GetPath()} with {len(points)} vertices and {len(indices)} faces."
                )
            else:
                mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
                wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=self.device)
                # print info
                omni.log.info(f"Created infinite plane mesh prim: {mesh_prim.GetPath()}.")
            # add the warp mesh to the list
            self.meshes[mesh_prim_path] = wp_mesh

        # throw an error if no meshes are found
        if all([mesh_prim_path not in self.meshes for mesh_prim_path in self.cfg.mesh_prim_paths]):
            raise RuntimeError(
                f"No meshes found for ray-casting! Please check the mesh prim paths: {self.cfg.mesh_prim_paths}"
            )

    def _initialize_rays_impl(self):
        # compute ray stars and directions
        self.ray_starts, self.ray_directions = self.cfg.pattern_cfg.func(self.cfg.pattern_cfg, self._device)
        self.num_rays = len(self.ray_directions)
        # apply offset transformation to the rays
        offset_pos = torch.tensor(list(self.cfg.offset.pos), device=self._device)
        offset_quat = torch.tensor(list(self.cfg.offset.rot), device=self._device)
        self.ray_directions = quat_apply(offset_quat.repeat(len(self.ray_directions), 1), self.ray_directions)
        self.ray_starts += offset_pos
        # repeat the rays for each sensor
        self.ray_starts = self.ray_starts.repeat(self._view.count, 1, 1)
        self.ray_directions = self.ray_directions.repeat(self._view.count, 1, 1)
        # prepare drift
        self.drift = torch.zeros(self._view.count, 3, device=self.device)
        self.ray_cast_drift = torch.zeros(self._view.count, 3, device=self.device)
        # fill the data buffer
        self._data.pos_w = torch.zeros(self._view.count, 3, device=self._device)
        self._data.quat_w = torch.zeros(self._view.count, 4, device=self._device)
        self._data.ray_hits_w = torch.zeros(self._view.count, self.num_rays, 3, device=self._device)
        self._data.ray_hits_b = torch.zeros(self._view.count, self.num_rays, 3, device=self._device)
        self._data.ray_hits_mask = torch.zeros(self._view.count, self.num_rays, dtype=torch.bool, device=self._device)

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""
        # obtain the poses of the sensors
        if isinstance(self._view, XFormPrim):
            pos_w, quat_w = self._view.get_world_poses(env_ids)
        elif isinstance(self._view, physx.ArticulationView):
            pos_w, quat_w = self._view.get_root_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = convert_quat(quat_w, to="wxyz")
        elif isinstance(self._view, physx.RigidBodyView):
            pos_w, quat_w = self._view.get_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = convert_quat(quat_w, to="wxyz")
        else:
            raise RuntimeError(f"Unsupported view type: {type(self._view)}")
        # note: we clone here because we are read-only operations
        pos_w = pos_w.clone()
        quat_w = quat_w.clone()
        # apply drift to ray starting position in world frame
        pos_w += self.drift[env_ids]
        # store the poses
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w[env_ids] = quat_w

        # check if user provided attach_yaw_only flag
        if self.cfg.attach_yaw_only is not None:
            msg = (
                "Raycaster attribute 'attach_yaw_only' property will be deprecated in a future release."
                " Please use the parameter 'ray_alignment' instead."
            )
            # set ray alignment to yaw
            if self.cfg.attach_yaw_only:
                self.cfg.ray_alignment = "yaw"
                msg += " Setting ray_alignment to 'yaw'."
            else:
                self.cfg.ray_alignment = "base"
                msg += " Setting ray_alignment to 'base'."
            # log the warning
            omni.log.warn(msg)

        # Update ray pattern if dynamic pattern is enabled
        if self.cfg.dynamic_pattern:
            # Re-compute ray starts and directions from pattern function
            new_ray_starts, new_ray_directions = self.cfg.pattern_cfg.func(self.cfg.pattern_cfg, self._device)
            # Apply offset transformation to the new rays
            offset_pos = torch.tensor(list(self.cfg.offset.pos), device=self._device)
            offset_quat = torch.tensor(list(self.cfg.offset.rot), device=self._device)
            new_ray_directions = quat_apply(offset_quat.repeat(len(new_ray_directions), 1), new_ray_directions)
            new_ray_starts += offset_pos
            # Update the sensor's ray starts and directions for all environments
            self.ray_starts[:, :len(new_ray_starts)] = new_ray_starts.unsqueeze(0).repeat(self._view.count, 1, 1)
            self.ray_directions[:, :len(new_ray_directions)] = new_ray_directions.unsqueeze(0).repeat(self._view.count, 1, 1)

        # ray cast based on the sensor poses
        if self.cfg.ray_alignment == "world":
            # apply horizontal drift to ray starting position in ray caster frame
            pos_w[:, 0:2] += self.ray_cast_drift[env_ids, 0:2]
            # no rotation is considered and directions are not rotated
            ray_starts_w = self.ray_starts[env_ids]
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = self.ray_directions[env_ids]
        elif self.cfg.ray_alignment == "yaw":
            # apply horizontal drift to ray starting position in ray caster frame
            pos_w[:, 0:2] += quat_apply_yaw(quat_w, self.ray_cast_drift[env_ids])[:, 0:2]
            # only yaw orientation is considered and directions are not rotated
            ray_starts_w = quat_apply_yaw(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = self.ray_directions[env_ids]
        elif self.cfg.ray_alignment == "base":
            # apply horizontal drift to ray starting position in ray caster frame
            pos_w[:, 0:2] += quat_apply(quat_w, self.ray_cast_drift[env_ids])[:, 0:2]
            # full orientation is considered
            ray_starts_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_directions[env_ids])
        else:
            raise RuntimeError(f"Unsupported ray_alignment type: {self.cfg.ray_alignment}.")

        # ray cast and store the hits
        # TODO: Make this work for multiple meshes?
        self._data.ray_hits_w[env_ids] = raycast_mesh(
            ray_starts_w,
            ray_directions_w,
            max_dist=self.cfg.max_distance,
            mesh=self.meshes[self.cfg.mesh_prim_paths[0]],
        )[0]

        # apply vertical drift to ray starting position in ray caster frame
        self._data.ray_hits_w[env_ids, :, 2] += self.ray_cast_drift[env_ids, 2].unsqueeze(-1)

        # 回归传感器本体坐标系（但采用水平基准）
        current_pos_w = self._data.pos_w[env_ids] # (N, 3) - 传感器在世界的位置
        current_quat_w = self._data.quat_w[env_ids] # (N, 4) - 传感器在世界的旋转 (wxyz)
        offset_pos = torch.tensor(list(self.cfg.offset.pos), device=self._device) # (3,)
        offset_pos_w = quat_apply(current_quat_w, offset_pos.unsqueeze(0).expand(len(env_ids), -1)) # (N, 3)
        hits_in_root_frame = self._data.ray_hits_w[env_ids] - current_pos_w.unsqueeze(1) # (N, B, 3)
        local_hits = hits_in_root_frame - offset_pos_w.unsqueeze(1) # (N, B, 3)

        if self.cfg.ray_alignment == "world":
            pass
        elif self.cfg.ray_alignment == "yaw":
            quat_yaw_inv = torch.stack([
                current_quat_w[:, 0],
                -current_quat_w[:, 1],
                -current_quat_w[:, 2],
                 torch.zeros_like(current_quat_w[:, 3])
            ], dim=-1)
            quat_yaw_inv = quat_yaw_inv / torch.linalg.norm(quat_yaw_inv, dim=-1, keepdim=True)
            local_hits = quat_apply_yaw(quat_yaw_inv.repeat(1, self.num_rays).view(-1, 4), local_hits.view(-1, 3)).view(local_hits.shape) # (N, B, 3)
        elif self.cfg.ray_alignment == "base":
            quat_inv = math_utils.quat_inv(current_quat_w) # (N, 4)
            local_hits = quat_apply(quat_inv.repeat(1, self.num_rays).view(-1, 4), local_hits.view(-1, 3)).view(local_hits.shape) # (N, B, 3)
        else:
            raise RuntimeError(f"Unsupported ray_alignment type: {self.cfg.ray_alignment}. Cannot transform to local frame.")

        # 包围盒裁剪
        if self.cfg.data_box_clip is not None:
            clip_bounds = torch.tensor(self.cfg.data_box_clip, dtype=local_hits.dtype, device=local_hits.device)
            half_clip_bounds = clip_bounds / 2.0 
            clip_mask = torch.all(torch.abs(local_hits) < half_clip_bounds, dim=-1) # (N, B)

            # 更新掩码：只有既不是 inf 又在包围盒内的点才是有效的
            hit_mask = ~torch.any(torch.isinf(self._data.ray_hits_w[env_ids]), dim=-1) # Shape: NxB
            final_mask = hit_mask & clip_mask # Shape: NxB (最终掩码)

            # 归一化处理
            if self.cfg.data_normalization: # 检查归一化参数是否存在且为True
                 normalized_local_hits = local_hits / clip_bounds # (N, B, 3)
                 local_hits = normalized_local_hits
        else:
            # 如果没有裁剪，则使用原始的 inf 掩码
            final_mask = ~torch.any(torch.isinf(self._data.ray_hits_w[env_ids]), dim=-1) # Shape: NxB

        self._data.ray_hits_b[env_ids] = local_hits
        self._data.ray_hits_mask[env_ids] = final_mask

        # data collection mode
        if self.cfg.data_collection:
            self.data_saver.save_data(self._data.ray_hits_b, self._data.ray_hits_mask)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            if not hasattr(self, "ray_visualizer"):
                self.ray_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            # set their visibility to true
            self.ray_visualizer.set_visibility(True)
        else:
            if hasattr(self, "ray_visualizer"):
                self.ray_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        viz_points = self._data.ray_hits_w[self._data.ray_hits_mask].view(-1, 3)
        # remove possible inf values
        viz_points = viz_points[~torch.any(torch.isinf(viz_points), dim=1)]
        if viz_points is not None and viz_points.shape[0] >= 1:
            # show ray hit positions
            self.ray_visualizer.visualize(viz_points)

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._view = None
