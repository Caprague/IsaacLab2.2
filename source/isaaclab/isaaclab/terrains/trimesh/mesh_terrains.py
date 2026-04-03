# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions to generate different terrains using the ``trimesh`` library."""

from __future__ import annotations

import numpy as np
import scipy.spatial.transform as tf
import torch
import trimesh
from typing import TYPE_CHECKING

from .utils import *  # noqa: F401, F403
from .utils import make_border, make_plane

if TYPE_CHECKING:
    from . import mesh_terrains_cfg


def flat_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshPlaneTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a flat terrain as a plane.

    .. image:: ../../_static/terrains/trimesh/flat_terrain.jpg
       :width: 45%
       :align: center

    Note:
        The :obj:`difficulty` parameter is ignored for this terrain.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # compute the position of the terrain
    origin = (cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.0)
    # compute the vertices of the terrain
    plane_mesh = make_plane(cfg.size, 0.0, center_zero=False)
    # return the tri-mesh and the position
    return [plane_mesh], np.array(origin)


def pyramid_stairs_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshPyramidStairsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a pyramid stair pattern.

    The terrain is a pyramid stair pattern which trims to a flat platform at the center of the terrain.

    If :obj:`cfg.holes` is True, the terrain will have pyramid stairs of length or width
    :obj:`cfg.platform_width` (depending on the direction) with no steps in the remaining area. Additionally,
    no border will be added.

    .. image:: ../../_static/terrains/trimesh/pyramid_stairs_terrain.jpg
       :width: 45%

    .. image:: ../../_static/terrains/trimesh/pyramid_stairs_terrain_with_holes.jpg
       :width: 45%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # resolve the terrain configuration
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])

    # compute number of steps in x and y direction
    num_steps_x = (cfg.size[0] - 2 * cfg.border_width - cfg.platform_width) // (2 * cfg.step_width) + 1
    num_steps_y = (cfg.size[1] - 2 * cfg.border_width - cfg.platform_width) // (2 * cfg.step_width) + 1
    # we take the minimum number of steps in x and y direction
    num_steps = int(min(num_steps_x, num_steps_y))

    # initialize list of meshes
    meshes_list = list()

    # generate the border if needed
    if cfg.border_width > 0.0 and not cfg.holes:
        # obtain a list of meshes for the border
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -step_height / 2]
        border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        make_borders = make_border(cfg.size, border_inner_size, step_height, border_center)
        # add the border meshes to the list of meshes
        meshes_list += make_borders

    # generate the terrain
    # -- compute the position of the center of the terrain
    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    terrain_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
    # -- generate the stair pattern
    for k in range(num_steps):
        # check if we need to add holes around the steps
        if cfg.holes:
            box_size = (cfg.platform_width, cfg.platform_width)
        else:
            box_size = (terrain_size[0] - 2 * k * cfg.step_width, terrain_size[1] - 2 * k * cfg.step_width)
        # compute the quantities of the box
        # -- location
        box_z = terrain_center[2] + k * step_height / 2.0
        box_offset = (k + 0.5) * cfg.step_width
        # -- dimensions
        box_height = (k + 2) * step_height
        # generate the boxes
        # top/bottom
        box_dims = (box_size[0], cfg.step_width, box_height)
        # -- top
        box_pos = (terrain_center[0], terrain_center[1] + terrain_size[1] / 2.0 - box_offset, box_z)
        box_top = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # -- bottom
        box_pos = (terrain_center[0], terrain_center[1] - terrain_size[1] / 2.0 + box_offset, box_z)
        box_bottom = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # right/left
        if cfg.holes:
            box_dims = (cfg.step_width, box_size[1], box_height)
        else:
            box_dims = (cfg.step_width, box_size[1] - 2 * cfg.step_width, box_height)
        # -- right
        box_pos = (terrain_center[0] + terrain_size[0] / 2.0 - box_offset, terrain_center[1], box_z)
        box_right = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # -- left
        box_pos = (terrain_center[0] - terrain_size[0] / 2.0 + box_offset, terrain_center[1], box_z)
        box_left = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # add the boxes to the list of meshes
        meshes_list += [box_top, box_bottom, box_right, box_left]

    # generate final box for the middle of the terrain
    box_dims = (
        terrain_size[0] - 2 * num_steps * cfg.step_width,
        terrain_size[1] - 2 * num_steps * cfg.step_width,
        (num_steps + 2) * step_height,
    )
    box_pos = (terrain_center[0], terrain_center[1], terrain_center[2] + num_steps * step_height / 2)
    box_middle = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    meshes_list.append(box_middle)

    if cfg.holes:
        # add a ground plane
        ground_plane = make_plane(cfg.size, height=0.0, center_zero=False)
        meshes_list.append(ground_plane)

    # origin of the terrain
    origin = np.array([terrain_center[0], terrain_center[1], (num_steps + 1) * step_height])

    return meshes_list, origin


def inverted_pyramid_stairs_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshInvertedPyramidStairsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a inverted pyramid stair pattern.

    The terrain is an inverted pyramid stair pattern which trims to a flat platform at the center of the terrain.

    If :obj:`cfg.holes` is True, the terrain will have pyramid stairs of length or width
    :obj:`cfg.platform_width` (depending on the direction) with no steps in the remaining area. Additionally,
    no border will be added.

    .. image:: ../../_static/terrains/trimesh/inverted_pyramid_stairs_terrain.jpg
       :width: 45%

    .. image:: ../../_static/terrains/trimesh/inverted_pyramid_stairs_terrain_with_holes.jpg
       :width: 45%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # resolve the terrain configuration
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])

    # compute number of steps in x and y direction
    num_steps_x = (cfg.size[0] - 2 * cfg.border_width - cfg.platform_width) // (2 * cfg.step_width) + 1
    num_steps_y = (cfg.size[1] - 2 * cfg.border_width - cfg.platform_width) // (2 * cfg.step_width) + 1
    # we take the minimum number of steps in x and y direction
    num_steps = int(min(num_steps_x, num_steps_y))
    # total height of the terrain
    total_height = (num_steps + 1) * step_height
    additional_height = 2.0

    # initialize list of meshes
    meshes_list = list()

    # generate the border if needed
    # if cfg.border_width > 0.0 and not cfg.holes:
    if cfg.border_width > 0.0:
        # obtain a list of meshes for the border
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -0.5 * step_height]
        border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        make_borders = make_border(cfg.size, border_inner_size, step_height, border_center)
        # add the border meshes to the list of meshes
        meshes_list += make_borders
    
        # Generate four boxes to cover the border area
        border_width = cfg.border_width
        terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
        box_height = total_height + additional_height

        # Top box
        top_box_dims = (cfg.size[0], border_width, box_height)
        top_box_pos = (terrain_center[0], terrain_center[1] + cfg.size[1] / 2 - border_width / 2, terrain_center[2] - box_height / 2)
        top_box = trimesh.creation.box(top_box_dims, trimesh.transformations.translation_matrix(top_box_pos))
        meshes_list.append(top_box)
 
        # Bottom box
        bottom_box_dims = (cfg.size[0], border_width, box_height)
        bottom_box_pos = (terrain_center[0], terrain_center[1] - cfg.size[1] / 2 + border_width / 2, terrain_center[2] - box_height / 2)
        bottom_box = trimesh.creation.box(bottom_box_dims, trimesh.transformations.translation_matrix(bottom_box_pos))
        meshes_list.append(bottom_box)

        # Right box
        right_box_dims = (border_width, cfg.size[1] - 2 * border_width, box_height)
        right_box_pos = (terrain_center[0] + cfg.size[0] / 2 - border_width / 2, terrain_center[1], terrain_center[2] - box_height / 2)
        right_box = trimesh.creation.box(right_box_dims, trimesh.transformations.translation_matrix(right_box_pos))
        meshes_list.append(right_box)

        # Left box
        left_box_dims = (border_width, cfg.size[1] - 2 * border_width, box_height)
        left_box_pos = (terrain_center[0] - cfg.size[0] / 2 + border_width / 2, terrain_center[1], terrain_center[2] - box_height / 2)
        left_box = trimesh.creation.box(left_box_dims, trimesh.transformations.translation_matrix(left_box_pos))
        meshes_list.append(left_box)

    # generate the terrain
    # -- compute the position of the center of the terrain
    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    terrain_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
    # -- generate the stair pattern
    for k in range(num_steps):
        # check if we need to add holes around the steps
        if cfg.holes:
            box_size = (cfg.platform_width, cfg.platform_width)
        else:
            box_size = (terrain_size[0] - 2 * k * cfg.step_width, terrain_size[1] - 2 * k * cfg.step_width)
        # compute the quantities of the box
        # -- location
        box_z = terrain_center[2] - total_height / 2 - (k + 1) * step_height / 2.0
        box_offset = (k + 0.5) * cfg.step_width
        # -- dimensions
        box_height = total_height - (k + 1) * step_height
        # generate the boxes
        # top/bottom
        box_dims = (box_size[0], cfg.step_width, box_height)
        # -- top
        box_pos = (terrain_center[0], terrain_center[1] + terrain_size[1] / 2.0 - box_offset, box_z)
        box_top = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # -- bottom
        box_pos = (terrain_center[0], terrain_center[1] - terrain_size[1] / 2.0 + box_offset, box_z)
        box_bottom = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # right/left
        if cfg.holes:
            box_dims = (cfg.step_width, box_size[1], box_height)
        else:
            box_dims = (cfg.step_width, box_size[1] - 2 * cfg.step_width, box_height)
        # -- right
        box_pos = (terrain_center[0] + terrain_size[0] / 2.0 - box_offset, terrain_center[1], box_z)
        box_right = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # -- left
        box_pos = (terrain_center[0] - terrain_size[0] / 2.0 + box_offset, terrain_center[1], box_z)
        box_left = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # add the boxes to the list of meshes
        meshes_list += [box_top, box_bottom, box_right, box_left]
    # generate final box for the middle of the terrain
    box_dims = (
        terrain_size[0] - 2 * num_steps * cfg.step_width,
        terrain_size[1] - 2 * num_steps * cfg.step_width,
        step_height,
    )
    box_pos = (terrain_center[0], terrain_center[1], terrain_center[2] - total_height - step_height / 2)
    box_middle = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    meshes_list.append(box_middle)
    
    # origin of the terrain
    origin = np.array([terrain_center[0], terrain_center[1], -(num_steps + 1) * step_height])

    if cfg.holes:
        # add a ground plane
        ground_plane = make_plane(cfg.size, height=-(total_height + additional_height), center_zero=False)
        meshes_list.append(ground_plane)

    return meshes_list, origin


def random_grid_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshRandomGridTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with cells of random heights and fixed width.

    The terrain is generated in the x-y plane and has a height of 1.0. It is then divided into a grid of the
    specified size :obj:`cfg.grid_width`. Each grid cell is then randomly shifted in the z-direction by a value uniformly
    sampled between :obj:`cfg.grid_height_range`. At the center of the terrain, a platform of the specified width
    :obj:`cfg.platform_width` is generated.

    If :obj:`cfg.holes` is True, the terrain will have randomized grid cells only along the plane extending
    from the platform (like a plus sign). The remaining area remains empty and no border will be added.

    .. image:: ../../_static/terrains/trimesh/random_grid_terrain.jpg
       :width: 45%

    .. image:: ../../_static/terrains/trimesh/random_grid_terrain_with_holes.jpg
       :width: 45%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).

    Raises:
        ValueError: If the terrain is not square. This method only supports square terrains.
        RuntimeError: If the grid width is large such that the border width is negative.
    """
    # check to ensure square terrain
    if cfg.size[0] != cfg.size[1]:
        raise ValueError(f"The terrain must be square. Received size: {cfg.size}.")
    # resolve the terrain configuration
    grid_height = cfg.grid_height_range[0] + difficulty * (cfg.grid_height_range[1] - cfg.grid_height_range[0])

    # initialize list of meshes
    meshes_list = list()
    # compute the number of boxes in each direction
    num_boxes_x = int(cfg.size[0] / cfg.grid_width)
    num_boxes_y = int(cfg.size[1] / cfg.grid_width)
    # constant parameters
    terrain_height = 1.0
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # generate the border
    border_width = cfg.border_width
    if border_width > 0:
        # compute parameters for the border
        border_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
        border_inner_size = (cfg.size[0] - border_width, cfg.size[1] - border_width)
        # create border meshes
        make_borders = make_border(cfg.size, border_inner_size, terrain_height, border_center)
        meshes_list += make_borders
    else:
        border_width = cfg.size[0] - min(num_boxes_x, num_boxes_y) * cfg.grid_width
        if border_width > 0:
            # compute parameters for the border
            border_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
            border_inner_size = (cfg.size[0] - border_width, cfg.size[1] - border_width)
            # create border meshes
            make_borders = make_border(cfg.size, border_inner_size, terrain_height, border_center)
            meshes_list += make_borders
        else:
            raise RuntimeError("Border width must be greater than 0! Adjust the parameter 'cfg.grid_width'.")
   
    # create a template grid of terrain height
    grid_dim = [cfg.grid_width, cfg.grid_width, terrain_height]
    grid_position = [0.5 * cfg.grid_width, 0.5 * cfg.grid_width, -terrain_height / 2]
    template_box = trimesh.creation.box(grid_dim, trimesh.transformations.translation_matrix(grid_position))
    # extract vertices and faces of the box to create a template
    template_vertices = template_box.vertices  # (8, 3)
    template_faces = template_box.faces

    # repeat the template box vertices to span the terrain (num_boxes_x * num_boxes_y, 8, 3)
    vertices = torch.tensor(template_vertices, device=device).repeat(num_boxes_x * num_boxes_y, 1, 1)
    # create a meshgrid to offset the vertices
    x = torch.arange(0, num_boxes_x, device=device)
    y = torch.arange(0, num_boxes_y, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    xx = xx.flatten().view(-1, 1)
    yy = yy.flatten().view(-1, 1)
    xx_yy = torch.cat((xx, yy), dim=1)
    # offset the vertices
    offsets = cfg.grid_width * xx_yy + border_width / 2
    vertices[:, :, :2] += offsets.unsqueeze(1)
    # mask the vertices to create holes, s.t. only grids along the x and y axis are present
    if cfg.holes:
        # -- x-axis
        mask_x = torch.logical_and(
            (vertices[:, :, 0] > (cfg.size[0] - border_width - cfg.platform_width) / 2).all(dim=1),
            (vertices[:, :, 0] < (cfg.size[0] + border_width + cfg.platform_width) / 2).all(dim=1),
        )
        vertices_x = vertices[mask_x]
        # -- y-axis
        mask_y = torch.logical_and(
            (vertices[:, :, 1] > (cfg.size[1] - border_width - cfg.platform_width) / 2).all(dim=1),
            (vertices[:, :, 1] < (cfg.size[1] + border_width + cfg.platform_width) / 2).all(dim=1),
        )
        vertices_y = vertices[mask_y]
        # -- combine these vertices
        vertices = torch.cat((vertices_x, vertices_y))
    # add noise to the vertices to have a random height over each grid cell
    num_boxes = len(vertices)
    # create noise for the z-axis
    h_noise = torch.zeros((num_boxes, 3), device=device)
    h_noise[:, 2].uniform_(-grid_height, grid_height)
    # reshape noise to match the vertices (num_boxes, 4, 3)
    # only the top vertices of the box are affected
    vertices_noise = torch.zeros((num_boxes, 4, 3), device=device)
    vertices_noise += h_noise.unsqueeze(1)
    # add height only to the top vertices of the box
    vertices[vertices[:, :, 2] == 0] += vertices_noise.view(-1, 3)
    # move to numpy
    vertices = vertices.reshape(-1, 3).cpu().numpy()

    # create faces for boxes (num_boxes, 12, 3). Each box has 6 faces, each face has 2 triangles.
    faces = torch.tensor(template_faces, device=device).repeat(num_boxes, 1, 1)
    face_offsets = torch.arange(0, num_boxes, device=device).unsqueeze(1).repeat(1, 12) * 8
    faces += face_offsets.unsqueeze(2)
    # move to numpy
    faces = faces.view(-1, 3).cpu().numpy()
    # convert to trimesh
    grid_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    meshes_list.append(grid_mesh)

    # add a platform in the center of the terrain that is accessible from all sides
    dim = (cfg.platform_width, cfg.platform_width, terrain_height + grid_height)
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2 + grid_height / 2)
    box_platform = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(box_platform)

    if cfg.holes:
        # add a ground plane
        ground_plane = make_plane(cfg.size, height=-1.0, center_zero=False)
        meshes_list.append(ground_plane)

    # specify the origin of the terrain
    origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], grid_height])

    return meshes_list, origin


def rails_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshRailsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with box rails as extrusions.

    The terrain contains two sets of box rails created as extrusions. The first set  (inner rails) is extruded from
    the platform at the center of the terrain, and the second set is extruded between the first set of rails
    and the terrain border. Each set of rails is extruded to the same height.

    .. image:: ../../_static/terrains/trimesh/rails_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. this is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # resolve the terrain configuration
    rail_height = cfg.rail_height_range[1] - difficulty * (cfg.rail_height_range[1] - cfg.rail_height_range[0])

    # initialize list of meshes
    meshes_list = list()
    # extract quantities
    rail_1_thickness, rail_2_thickness = cfg.rail_thickness_range
    rail_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], rail_height * 0.5)
    # constants for terrain generation
    terrain_height = 1.0
    rail_2_ratio = 0.6

    # generate first set of rails
    rail_1_inner_size = (cfg.platform_width, cfg.platform_width)
    rail_1_outer_size = (cfg.platform_width + 2.0 * rail_1_thickness, cfg.platform_width + 2.0 * rail_1_thickness)
    meshes_list += make_border(rail_1_outer_size, rail_1_inner_size, rail_height, rail_center)
    # generate second set of rails
    rail_2_inner_x = cfg.platform_width + (cfg.size[0] - cfg.platform_width) * rail_2_ratio
    rail_2_inner_y = cfg.platform_width + (cfg.size[1] - cfg.platform_width) * rail_2_ratio
    rail_2_inner_size = (rail_2_inner_x, rail_2_inner_y)
    rail_2_outer_size = (rail_2_inner_x + 2.0 * rail_2_thickness, rail_2_inner_y + 2.0 * rail_2_thickness)
    meshes_list += make_border(rail_2_outer_size, rail_2_inner_size, rail_height, rail_center)
    # generate the ground
    dim = (cfg.size[0], cfg.size[1], terrain_height)
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
    ground_meshes = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(ground_meshes)

    # specify the origin of the terrain
    origin = np.array([pos[0], pos[1], 0.0])

    return meshes_list, origin


def pit_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshPitTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a pit with levels (stairs) leading out of the pit.

    The terrain contains a platform at the center and a staircase leading out of the pit.
    The staircase is a series of steps that are aligned along the x- and y- axis. The steps are
    created by extruding a ring along the x- and y- axis. If :obj:`is_double_pit` is True, the pit
    contains two levels.

    .. image:: ../../_static/terrains/trimesh/pit_terrain.jpg
       :width: 40%

    .. image:: ../../_static/terrains/trimesh/pit_terrain_with_two_levels.jpg
       :width: 40%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # resolve the terrain configuration
    pit_depth = cfg.pit_depth_range[0] + difficulty * (cfg.pit_depth_range[1] - cfg.pit_depth_range[0])

    # initialize list of meshes
    meshes_list = list()
    # extract quantities
    inner_pit_size = (cfg.platform_width, cfg.platform_width)
    total_depth = pit_depth
    # constants for terrain generation
    terrain_height = 1.0
    ring_2_ratio = 0.6

    # if the pit is double, the inner ring is smaller to fit the second level
    if cfg.double_pit:
        # increase the total height of the pit
        total_depth *= 2.0
        # reduce the size of the inner ring
        inner_pit_x = cfg.platform_width + (cfg.size[0] - cfg.platform_width) * ring_2_ratio
        inner_pit_y = cfg.platform_width + (cfg.size[1] - cfg.platform_width) * ring_2_ratio
        inner_pit_size = (inner_pit_x, inner_pit_y)

    # generate the pit (outer ring)
    pit_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -total_depth * 0.5]
    meshes_list += make_border(cfg.size, inner_pit_size, total_depth, pit_center)
    # generate the second level of the pit (inner ring)
    if cfg.double_pit:
        pit_center[2] = -total_depth
        meshes_list += make_border(inner_pit_size, (cfg.platform_width, cfg.platform_width), total_depth, pit_center)
    # generate the ground
    dim = (cfg.size[0], cfg.size[1], terrain_height)
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -total_depth - terrain_height / 2)
    ground_meshes = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(ground_meshes)

    # specify the origin of the terrain
    origin = np.array([pos[0], pos[1], -total_depth])

    return meshes_list, origin


def box_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshBoxTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with boxes (similar to a pyramid).

    The terrain has a ground with boxes on top of it that are stacked on top of each other.
    The boxes are created by extruding a rectangle along the z-axis. If :obj:`double_box` is True,
    then two boxes of height :obj:`box_height` are stacked on top of each other.

    .. image:: ../../_static/terrains/trimesh/box_terrain.jpg
       :width: 40%

    .. image:: ../../_static/terrains/trimesh/box_terrain_with_two_boxes.jpg
       :width: 40%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # resolve the terrain configuration
    box_height = cfg.box_height_range[0] + difficulty * (cfg.box_height_range[1] - cfg.box_height_range[0])

    # initialize list of meshes
    meshes_list = list()
    # extract quantities
    total_height = box_height
    if cfg.double_box:
        total_height *= 2.0
    # constants for terrain generation
    terrain_height = 1.0
    box_2_ratio = 0.6

    # Generate the top box
    dim = (cfg.platform_width, cfg.platform_width, terrain_height + total_height)
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], (total_height - terrain_height) / 2)
    box_mesh = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(box_mesh)
    # Generate the lower box
    if cfg.double_box:
        # calculate the size of the lower box
        outer_box_x = cfg.platform_width + (cfg.size[0] - cfg.platform_width) * box_2_ratio
        outer_box_y = cfg.platform_width + (cfg.size[1] - cfg.platform_width) * box_2_ratio
        # create the lower box
        dim = (outer_box_x, outer_box_y, terrain_height + total_height / 2)
        pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], (total_height - terrain_height) / 2 - total_height / 4)
        box_mesh = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
        meshes_list.append(box_mesh)
    # Generate the ground
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
    dim = (cfg.size[0], cfg.size[1], terrain_height)
    ground_mesh = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(ground_mesh)

    # specify the origin of the terrain
    origin = np.array([pos[0], pos[1], total_height])

    return meshes_list, origin


def gap_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshGapTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a gap around the platform.

    The terrain has a ground with a platform in the middle. The platform is surrounded by a gap
    of width :obj:`gap_width` on all sides.

    .. image:: ../../_static/terrains/trimesh/gap_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # resolve the terrain configuration
    gap_width = cfg.gap_width_range[0] + difficulty * (cfg.gap_width_range[1] - cfg.gap_width_range[0])

    # initialize list of meshes
    meshes_list = list()
    # constants for terrain generation
    terrain_height = 1.0
    terrain_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)

    # Generate the outer ring
    inner_size = (cfg.platform_width + 2 * gap_width, cfg.platform_width + 2 * gap_width)
    meshes_list += make_border(cfg.size, inner_size, terrain_height, terrain_center)
    # Generate the inner box
    box_dim = (cfg.platform_width, cfg.platform_width, terrain_height)
    box = trimesh.creation.box(box_dim, trimesh.transformations.translation_matrix(terrain_center))
    meshes_list.append(box)
    # Add a ground plane
    ground_plane = make_plane(cfg.size, height=-1.0, center_zero=False)
    meshes_list.append(ground_plane)

    # specify the origin of the terrain
    origin = np.array([terrain_center[0], terrain_center[1], 0.0])

    return meshes_list, origin


def floating_ring_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshFloatingRingTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a floating square ring.

    The terrain has a ground with a floating ring in the middle. The ring extends from the center from
    :obj:`platform_width` to :obj:`platform_width` + :obj:`ring_width` in the x and y directions.
    The thickness of the ring is :obj:`ring_thickness` and the height of the ring from the terrain
    is :obj:`ring_height`.

    .. image:: ../../_static/terrains/trimesh/floating_ring_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # resolve the terrain configuration
    ring_height = cfg.ring_height_range[1] - difficulty * (cfg.ring_height_range[1] - cfg.ring_height_range[0])
    ring_width = cfg.ring_width_range[0] + difficulty * (cfg.ring_width_range[1] - cfg.ring_width_range[0])

    # initialize list of meshes
    meshes_list = list()
    # constants for terrain generation
    terrain_height = 1.0

    # Generate the floating ring
    ring_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], ring_height + 0.5 * cfg.ring_thickness)
    ring_outer_size = (cfg.platform_width + 2 * ring_width, cfg.platform_width + 2 * ring_width)
    ring_inner_size = (cfg.platform_width, cfg.platform_width)
    meshes_list += make_border(ring_outer_size, ring_inner_size, cfg.ring_thickness, ring_center)
    # Generate the ground
    dim = (cfg.size[0], cfg.size[1], terrain_height)
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
    ground = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(ground)

    # specify the origin of the terrain
    origin = np.asarray([pos[0], pos[1], 0.0])

    return meshes_list, origin


def star_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshStarTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a star.

    The terrain has a ground with a cylinder in the middle. The star is made of :obj:`num_bars` bars
    with a width of :obj:`bar_width` and a height of :obj:`bar_height`. The bars are evenly
    spaced around the cylinder and connect to the peripheral of the terrain.

    .. image:: ../../_static/terrains/trimesh/star_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).

    Raises:
        ValueError: If :obj:`num_bars` is less than 2.
    """
    # check the number of bars
    if cfg.num_bars < 2:
        raise ValueError(f"The number of bars in the star must be greater than 2. Received: {cfg.num_bars}")

    # resolve the terrain configuration
    bar_height = cfg.bar_height_range[0] + difficulty * (cfg.bar_height_range[1] - cfg.bar_height_range[0])
    bar_width = cfg.bar_width_range[1] - difficulty * (cfg.bar_width_range[1] - cfg.bar_width_range[0])

    # initialize list of meshes
    meshes_list = list()
    # Generate a platform in the middle
    platform_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -bar_height / 2)
    platform_transform = trimesh.transformations.translation_matrix(platform_center)
    platform = trimesh.creation.cylinder(
        cfg.platform_width * 0.5, bar_height, sections=2 * cfg.num_bars, transform=platform_transform
    )
    meshes_list.append(platform)
    # Generate bars to connect the platform to the terrain
    transform = np.eye(4)
    transform[:3, -1] = np.asarray(platform_center)
    yaw = 0.0
    for _ in range(cfg.num_bars):
        # compute the length of the bar based on the yaw
        # length changes since the bar is connected to a square border
        bar_length = cfg.size[0]
        if yaw < 0.25 * np.pi:
            bar_length /= np.math.cos(yaw)
        elif yaw < 0.75 * np.pi:
            bar_length /= np.math.sin(yaw)
        else:
            bar_length /= np.math.cos(np.pi - yaw)
        # compute the transform of the bar
        transform[0:3, 0:3] = tf.Rotation.from_euler("z", yaw).as_matrix()
        # add the bar to the mesh
        dim = [bar_length - bar_width, bar_width, bar_height]
        bar = trimesh.creation.box(dim, transform)
        meshes_list.append(bar)
        # increment the yaw
        yaw += np.pi / cfg.num_bars
    # Generate the exterior border
    inner_size = (cfg.size[0] - 2 * bar_width, cfg.size[1] - 2 * bar_width)
    meshes_list += make_border(cfg.size, inner_size, bar_height, platform_center)
    # Generate the ground
    ground = make_plane(cfg.size, -bar_height, center_zero=False)
    meshes_list.append(ground)
    # specify the origin of the terrain
    origin = np.asarray([0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0])

    return meshes_list, origin


def star_inv_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshStarInvTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with an inverted star (star-shaped protrusion).

    The terrain has a ground with a cylinder in the middle. The inverted star is made of :obj:`num_bars` bars
    with a width of :obj:`bar_width` and a height of :obj:`bar_height`. The bars are evenly
    spaced around the cylinder and connect to the peripheral of the terrain.

    .. image:: ../../_static/terrains/trimesh/star_inv_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).

    Raises:
        ValueError: If :obj:`num_bars` is less than 2.
    """
    # check the number of bars
    if cfg.num_bars < 2:
        raise ValueError(f"The number of bars in the star must be greater than 2. Received: {cfg.num_bars}")

    # resolve the terrain configuration
    bar_height = cfg.bar_height_range[0] + difficulty * (cfg.bar_height_range[1] - cfg.bar_height_range[0])
    bar_width = cfg.bar_width_range[1] - difficulty * (cfg.bar_width_range[1] - cfg.bar_width_range[0])

    # initialize list of meshes
    meshes_list = list()

    # Create the large raised terrain block
    terrain_block_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], bar_height / 2)
    terrain_block = trimesh.creation.box(
        [cfg.size[0], cfg.size[1], bar_height],
        trimesh.transformations.translation_matrix(terrain_block_center)
    )

    # Generate the star-shaped area to be carved out
    # Generate a platform in the middle
    platform_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], bar_height / 2)
    platform_transform = trimesh.transformations.translation_matrix(platform_center)
    platform = trimesh.creation.cylinder(
        cfg.platform_width * 0.5, bar_height, sections=2 * cfg.num_bars, transform=platform_transform
    )

    # Generate bars to connect the platform to the terrain
    star_meshes = [platform]
    transform = np.eye(4)
    transform[:3, -1] = np.asarray(platform_center)
    yaw = 0.0
    for _ in range(cfg.num_bars):
        # compute the length of the bar based on the yaw
        # length changes since the bar is connected to a square border
        bar_length = cfg.size[0]
        if yaw < 0.25 * np.pi:
            bar_length /= np.math.cos(yaw)
        elif yaw < 0.75 * np.pi:
            bar_length /= np.math.sin(yaw)
        else:
            bar_length /= np.math.cos(np.pi - yaw)
        # compute the transform of the bar
        transform[0:3, 0:3] = tf.Rotation.from_euler("z", yaw).as_matrix()
        # add the bar to the mesh
        dim = [bar_length - bar_width, bar_width, bar_height]
        bar = trimesh.creation.box(dim, transform)
        star_meshes.append(bar)
        # increment the yaw
        yaw += np.pi / cfg.num_bars

    # Subtract each star mesh from the terrain block one by one
    carved_terrain = terrain_block
    for mesh in star_meshes:
        carved_terrain = carved_terrain.difference(mesh)

    # Generate the exterior border using four boxes
    inner_size = (cfg.size[0] - 2 * bar_width, cfg.size[1] - 2 * bar_width)
    # Top box
    top_box_dim = [cfg.size[0], bar_width, bar_height]
    top_box_center = (0.5 * cfg.size[0], cfg.size[1] - bar_width / 2, bar_height / 2)
    top_box_transform = trimesh.transformations.translation_matrix(top_box_center)
    top_box = trimesh.creation.box(top_box_dim, top_box_transform)
    carved_terrain = carved_terrain.difference(top_box)

    # Bottom box
    bottom_box_dim = [cfg.size[0], bar_width, bar_height]
    bottom_box_center = (0.5 * cfg.size[0], bar_width / 2, bar_height / 2)
    bottom_box_transform = trimesh.transformations.translation_matrix(bottom_box_center)
    bottom_box = trimesh.creation.box(bottom_box_dim, bottom_box_transform)
    carved_terrain = carved_terrain.difference(bottom_box)

    # Right box
    right_box_dim = [bar_width, inner_size[1], bar_height]
    right_box_center = (cfg.size[0] - bar_width / 2, 0.5 * cfg.size[1], bar_height / 2)
    right_box_transform = trimesh.transformations.translation_matrix(right_box_center)
    right_box = trimesh.creation.box(right_box_dim, right_box_transform)
    carved_terrain = carved_terrain.difference(right_box)

    # Left box
    left_box_dim = [bar_width, inner_size[1], bar_height]
    left_box_center = (bar_width / 2, 0.5 * cfg.size[1], bar_height / 2)
    left_box_transform = trimesh.transformations.translation_matrix(left_box_center)
    left_box = trimesh.creation.box(left_box_dim, left_box_transform)
    carved_terrain = carved_terrain.difference(left_box)

    meshes_list.append(carved_terrain)

    # Generate the ground
    ground = make_plane(cfg.size, 0, center_zero=False)
    meshes_list.append(ground)

    # specify the origin of the terrain
    origin = np.asarray([0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0])

    return meshes_list, origin


def cross_obstacle_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshCrossObstacleTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    # resolve the terrain configuration
    cross_length = cfg.cross_length_range[0] + difficulty * (cfg.cross_length_range[1] - cfg.cross_length_range[0])
    cross_width = np.random.uniform(cfg.cross_width_range[0], cfg.cross_width_range[1])
    cross_height = np.random.uniform(cfg.cross_height_range[0], cfg.cross_height_range[1])

    # initialize list of meshes
    meshes_list = list()

    cross_width = np.random.uniform(cfg.cross_width_range[0], cfg.cross_width_range[1])
    cross_height = np.random.uniform(cfg.cross_height_range[0], cfg.cross_height_range[1])
    box_dim = [cross_length, cross_width, cross_height]
    box_center = (0.25 * cfg.size[0], 0.25 * cfg.size[1], cross_height / 2)
    box_transform = trimesh.transformations.translation_matrix(box_center)
    box = trimesh.creation.box(box_dim, box_transform)
    meshes_list.append(box)
    box_dim = [cross_width, cross_length, cross_height]
    box = trimesh.creation.box(box_dim, box_transform)
    meshes_list.append(box)


    cross_width = np.random.uniform(cfg.cross_width_range[0], cfg.cross_width_range[1])
    cross_height = np.random.uniform(cfg.cross_height_range[0], cfg.cross_height_range[1])
    box_dim = [cross_length, cross_width, cross_height]
    box_center = (0.75 * cfg.size[0], 0.25 * cfg.size[1], cross_height / 2)
    box_transform = trimesh.transformations.translation_matrix(box_center)
    box = trimesh.creation.box(box_dim, box_transform)
    meshes_list.append(box)
    box_dim = [cross_width, cross_length, cross_height]
    box = trimesh.creation.box(box_dim, box_transform)
    meshes_list.append(box)

    cross_width = np.random.uniform(cfg.cross_width_range[0], cfg.cross_width_range[1])
    cross_height = np.random.uniform(cfg.cross_height_range[0], cfg.cross_height_range[1])
    box_dim = [cross_length, cross_width, cross_height]
    box_center = (0.25 * cfg.size[0], 0.75 * cfg.size[1], cross_height / 2)
    box_transform = trimesh.transformations.translation_matrix(box_center)
    box = trimesh.creation.box(box_dim, box_transform)
    meshes_list.append(box)
    box_dim = [cross_width, cross_length, cross_height]
    box = trimesh.creation.box(box_dim, box_transform)
    meshes_list.append(box)

    cross_width = np.random.uniform(cfg.cross_width_range[0], cfg.cross_width_range[1])
    cross_height = np.random.uniform(cfg.cross_height_range[0], cfg.cross_height_range[1])
    box_dim = [cross_length, cross_width, cross_height]
    box_center = (0.75 * cfg.size[0], 0.75 * cfg.size[1], cross_height / 2)
    box_transform = trimesh.transformations.translation_matrix(box_center)
    box = trimesh.creation.box(box_dim, box_transform)
    meshes_list.append(box)
    box_dim = [cross_width, cross_length, cross_height]
    box = trimesh.creation.box(box_dim, box_transform)
    meshes_list.append(box)

    # Generate the ground
    ground = make_plane(cfg.size, 0, center_zero=False)
    meshes_list.append(ground)

    # specify the origin of the terrain
    origin = np.asarray([0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0])

    return meshes_list, origin


def pallets_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshPalletsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with pallets (multiple square ring platforms at different heights).

    The terrain has a central platform with a deep pit around it. Inside the pit, there are multiple
    square ring platforms distributed at different distances, with varying widths and heights.

    .. image:: ../../_static/terrains/trimesh/pallets_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the list of tri-meshes of the terrain and the origin of the terrain (in m).
    """
    # Initialize list of meshes
    meshes_list = list()
    # Constants for terrain generation
    terrain_height = 1.0
    
    # Calculate terrain center
    terrain_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0)
    
    # Calculate the available space for rings
    available_space = min(cfg.size[0], cfg.size[1]) - cfg.platform_width
    if cfg.border_width > 0.0:
        available_space -= 2 * cfg.border_width
    available_space /= 2  # Since we're working from the center
    
    # Generate the central platform (上平面高度为0)
    platform_dim = (cfg.platform_width, cfg.platform_width, terrain_height)
    # 平台中心z坐标设为terrain_height/2，确保上平面高度为0
    platform_pos = (terrain_center[0], terrain_center[1], 0.0 - terrain_height / 2)
    platform = trimesh.creation.box(platform_dim, trimesh.transformations.translation_matrix(platform_pos))
    meshes_list.append(platform)
    
    # Generate multiple square ring platforms until space is filled
    current_radius = cfg.platform_width / 2  # 从中心平台边缘开始
    ring_index = 0
    
    # 计算基于difficulty的宽度和间隔参数权重
    width_weight = 1.0 - difficulty  # 宽度权重：高难度时大
    spacing_weight = difficulty  # 间隔权重：高难度时大
    
    # 确保至少生成一个环
    has_generated_any_ring = False
    
    # 生成环的策略调整：使用更积极的方式填充空间
    while True:
        # 计算基于difficulty的环间距范围
        spacing_min = cfg.ring_spacing_range[0]
        spacing_max = cfg.ring_spacing_range[1]
        # 低难度时，间距更偏向最小值；高难度时，更偏向最大值
        spacing_mid = spacing_min + spacing_weight * (spacing_max - spacing_min)
        
        if cfg.randomize_widths:
            # 基于difficulty的随机间距：低难度时更接近最小值
            ring_spacing = np.random.normal(spacing_mid, (spacing_max - spacing_min) / 6)
            ring_spacing = np.clip(ring_spacing, spacing_min, spacing_max)
        else:
            # 直接使用中间值
            ring_spacing = spacing_mid
        
        # 移动到下一个环的位置
        current_radius += ring_spacing
        
        # 计算基于difficulty的环宽度范围
        width_min = cfg.ring_width_range[0]
        width_max = cfg.ring_width_range[1]
        # 修正：低难度时，宽度更偏向最小值；高难度时，更偏向最大值
        width_mid = width_min + width_weight * (width_max - width_min)
        
        if cfg.randomize_widths:
            # 基于difficulty的随机宽度：低难度时更接近最小值
            ring_width = np.random.normal(width_mid, (width_max - width_min) / 6)
            ring_width = np.clip(ring_width, width_min, width_max)
        else:
            # 直接使用中间值
            ring_width = width_mid
        
        # 检查是否还有足够空间放置这个环
        remaining_space = available_space - current_radius
        
        # 如果剩余空间不足，但我们还没有生成任何环，就尝试调整宽度以适应
        if remaining_space <= 0:
            if not has_generated_any_ring and available_space > current_radius - ring_spacing:
                # 尝试创建一个尽可能宽的环来填充剩余空间
                ring_width = max(0.1, available_space - (current_radius - ring_spacing))
                current_radius = current_radius - ring_spacing
            else:
                break
        
        # 计算环高度
        if cfg.randomize_heights:
            # 随机高度，以0为基准
            ring_height = np.random.uniform(cfg.ring_height_range[0], cfg.ring_height_range[1])
        else:
            # 使用正弦函数创建有规律的分布，以0为基准
            t = (np.sin(ring_index * 0.5) + 1.0) * 0.5 - 0.5  # 范围在[-0.5, 0.5]
            # 应用难度影响：低难度时更接近0，高难度时变化更大
            t *= difficulty
            # 映射到配置的高度范围
            ring_height = t * 2 * max(abs(cfg.ring_height_range[0]), abs(cfg.ring_height_range[1]))
            ring_height = np.clip(ring_height, cfg.ring_height_range[0], cfg.ring_height_range[1])
        
        # 计算环尺寸
        ring_outer_size = (2 * (current_radius + ring_width), 2 * (current_radius + ring_width))
        ring_inner_size = (2 * current_radius, 2 * current_radius)
        # 环的中心z坐标以0为基准，加上ring_height
        ring_center = (terrain_center[0], terrain_center[1], ring_height - cfg.ring_thickness/2.0)
        
        # 生成环
        ring_meshes = make_border(ring_outer_size, ring_inner_size, cfg.ring_thickness, ring_center)
        meshes_list += ring_meshes
        
        has_generated_any_ring = True
        
        # 更新当前半径到环的外边缘
        current_radius += ring_width
        ring_index += 1
        
        # 添加额外检查，防止无限循环
        if ring_index > 50:  # 限制最大环数
            break
    
    # 计算实际的最外环半径
    actual_outer_radius = current_radius - ring_spacing
    
    # 计算理论上的最大可用半径
    theoretical_max_radius = min(cfg.size[0], cfg.size[1]) / 2
    if cfg.border_width > 0.0:
        theoretical_max_radius -= cfg.border_width
    
    # 计算外围空白区域宽度
    outer_gap = theoretical_max_radius - actual_outer_radius
    
    # 确定自适应的border_width
    adaptive_border_width = cfg.border_width
    # 如果外围空白区域超过阈值，则增加border_width来填充
    if outer_gap > cfg.border_fill_threshold:
        adaptive_border_width = min(cfg.max_border_width, cfg.border_width + outer_gap)
    
    # 生成自适应的外边框
    if adaptive_border_width > 0.0:
        border_size = (cfg.size[0], cfg.size[1])
        inner_size = (cfg.size[0] - 2 * adaptive_border_width, cfg.size[1] - 2 * adaptive_border_width)
        border_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -cfg.pit_depth / 2)
        meshes_list += make_border(border_size, inner_size, cfg.pit_depth, border_center)
    
    # 生成地面（坑的底部）
    ground_dim = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width, terrain_height)
    ground_pos = (terrain_center[0], terrain_center[1], -cfg.pit_depth - terrain_height / 2)
    ground = trimesh.creation.box(ground_dim, trimesh.transformations.translation_matrix(ground_pos))
    meshes_list.append(ground)
    
    # 指定地形的原点
    origin = np.array([terrain_center[0], terrain_center[1], 0.0])
    
    return meshes_list, origin


def repeated_objects_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshRepeatedObjectsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a set of repeated objects.

    The terrain has a ground with a platform in the middle. The objects are randomly placed on the
    terrain s.t. they do not overlap with the platform.

    Depending on the object type, the objects are generated with different parameters. The objects
    The types of objects that can be generated are: ``"cylinder"``, ``"box"``, ``"cone"``.

    The object parameters are specified in the configuration as curriculum parameters. The difficulty
    is used to linearly interpolate between the minimum and maximum values of the parameters.

    .. image:: ../../_static/terrains/trimesh/repeated_objects_cylinder_terrain.jpg
       :width: 30%

    .. image:: ../../_static/terrains/trimesh/repeated_objects_box_terrain.jpg
       :width: 30%

    .. image:: ../../_static/terrains/trimesh/repeated_objects_pyramid_terrain.jpg
       :width: 30%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).

    Raises:
        ValueError: If the object type is not supported. It must be either a string or a callable.
    """
    # import the object functions -- this is done here to avoid circular imports
    from .mesh_terrains_cfg import (
        MeshRepeatedBoxesTerrainCfg,
        MeshRepeatedCylindersTerrainCfg,
        MeshRepeatedPyramidsTerrainCfg,
    )

    # if object type is a string, get the function: make_{object_type}
    if isinstance(cfg.object_type, str):
        object_func = globals().get(f"make_{cfg.object_type}")
    else:
        object_func = cfg.object_type
    if not callable(object_func):
        raise ValueError(f"The attribute 'object_type' must be a string or a callable. Received: {object_func}")

    # Resolve the terrain configuration
    # -- pass parameters to make calling simpler
    cp_0 = cfg.object_params_start
    cp_1 = cfg.object_params_end
    # -- common parameters
    num_objects = cp_0.num_objects + int(difficulty * (cp_1.num_objects - cp_0.num_objects))
    height = cp_0.height + difficulty * (cp_1.height - cp_0.height)
    platform_height = cfg.platform_height if cfg.platform_height >= 0.0 else height
    # -- object specific parameters
    # note: SIM114 requires duplicated logical blocks under a single body.
    if isinstance(cfg, MeshRepeatedBoxesTerrainCfg):
        cp_0: MeshRepeatedBoxesTerrainCfg.ObjectCfg
        cp_1: MeshRepeatedBoxesTerrainCfg.ObjectCfg
        object_kwargs = {
            "length": cp_0.size[0] + difficulty * (cp_1.size[0] - cp_0.size[0]),
            "width": cp_0.size[1] + difficulty * (cp_1.size[1] - cp_0.size[1]),
            "max_yx_angle": cp_0.max_yx_angle + difficulty * (cp_1.max_yx_angle - cp_0.max_yx_angle),
            "degrees": cp_0.degrees,
        }
    elif isinstance(cfg, MeshRepeatedPyramidsTerrainCfg):  # noqa: SIM114
        cp_0: MeshRepeatedPyramidsTerrainCfg.ObjectCfg
        cp_1: MeshRepeatedPyramidsTerrainCfg.ObjectCfg
        object_kwargs = {
            "radius": cp_0.radius + difficulty * (cp_1.radius - cp_0.radius),
            "max_yx_angle": cp_0.max_yx_angle + difficulty * (cp_1.max_yx_angle - cp_0.max_yx_angle),
            "degrees": cp_0.degrees,
        }
    elif isinstance(cfg, MeshRepeatedCylindersTerrainCfg):  # noqa: SIM114
        cp_0: MeshRepeatedCylindersTerrainCfg.ObjectCfg
        cp_1: MeshRepeatedCylindersTerrainCfg.ObjectCfg
        object_kwargs = {
            "radius": cp_0.radius + difficulty * (cp_1.radius - cp_0.radius),
            "max_yx_angle": cp_0.max_yx_angle + difficulty * (cp_1.max_yx_angle - cp_0.max_yx_angle),
            "degrees": cp_0.degrees,
        }
    else:
        raise ValueError(f"Unknown terrain configuration: {cfg}")
    # constants for the terrain
    platform_clearance = 0.1
    # 定义地形边缘的安全距离
    terrain_edge_clearance = cfg.terrain_edge_clearance

    # initialize list of meshes
    meshes_list = list()
    # compute quantities
    origin = np.asarray((0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.5 * platform_height))
    platform_corners = np.asarray([
        [origin[0] - cfg.platform_width / 2, origin[1] - cfg.platform_width / 2],
        [origin[0] + cfg.platform_width / 2, origin[1] + cfg.platform_width / 2],
    ])
    platform_corners[0, :] *= 1 - platform_clearance
    platform_corners[1, :] *= 1 + platform_clearance
    # sample valid center for objects
    object_centers = np.zeros((num_objects, 3))
    # use a mask to track invalid objects that still require sampling
    mask_objects_left = np.ones((num_objects,), dtype=bool)
    # loop until no objects are left to sample
    while np.any(mask_objects_left):
        # only sample the centers of the remaining invalid objects
        num_objects_left = mask_objects_left.sum()
        object_centers[mask_objects_left, 0] = np.random.uniform(0, cfg.size[0], num_objects_left)
        object_centers[mask_objects_left, 1] = np.random.uniform(0, cfg.size[1], num_objects_left)
        # filter out the centers that are on the platform
        is_within_platform_x = np.logical_and(
            object_centers[mask_objects_left, 0] >= platform_corners[0, 0],
            object_centers[mask_objects_left, 0] <= platform_corners[1, 0],
        )
        is_within_platform_y = np.logical_and(
            object_centers[mask_objects_left, 1] >= platform_corners[0, 1],
            object_centers[mask_objects_left, 1] <= platform_corners[1, 1],
        )
        # update the mask to track the validity of the objects sampled in this iteration
        mask_objects_left[mask_objects_left] = np.logical_and(is_within_platform_x, is_within_platform_y)

    # generate obstacles (but keep platform clean)
    for index in range(len(object_centers)):
        # randomize the height of the object
        abs_height_noise = np.random.uniform(cfg.abs_height_noise[0], cfg.abs_height_noise[1])
        rel_height_noise = np.random.uniform(cfg.rel_height_noise[0], cfg.rel_height_noise[1])
        ob_height = height * rel_height_noise + abs_height_noise
        if ob_height > 0.0:
            object_mesh = object_func(center=object_centers[index], height=ob_height, **object_kwargs)
            meshes_list.append(object_mesh)

    # 移除地形边缘 1.0 米内的障碍物
    valid_meshes = []
    for index, obj_center in enumerate(object_centers):
        # 检查对象中心是否在地形边缘 1.0 米内
        if (
            obj_center[0] >= terrain_edge_clearance and
            obj_center[0] <= cfg.size[0] - terrain_edge_clearance and
            obj_center[1] >= terrain_edge_clearance and
            obj_center[1] <= cfg.size[1] - terrain_edge_clearance
        ):
            valid_meshes.append(meshes_list[index])
    meshes_list = valid_meshes

    # generate a ground plane for the terrain
    ground_plane = make_plane(cfg.size, height=0.0, center_zero=False)
    meshes_list.append(ground_plane)
    # # generate a platform in the middle
    # dim = (cfg.platform_width, cfg.platform_width, 0.5 * platform_height)
    # pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.25 * platform_height)
    # platform = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    # meshes_list.append(platform)

    return meshes_list, origin


def mesh_stepping_stones_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshSteppingStonesTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a stepping stones terrain using mesh objects (boxes and cylinders).
    
    The terrain has a ground plane with holes (pits) and stepping stones (boxes and cylinders)
    placed in a grid-like pattern. A platform is placed in the center of the terrain.
    
    The stepping stones are placed such that they form a path that the agent can traverse.
    The stones can be slightly tilted to increase the challenge.
    
    .. image:: ../../_static/terrains/trimesh/mesh_stepping_stones_terrain.jpg
       :width: 40%
       :align: center
    
    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.
    
    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # Resolve the terrain configuration based on difficulty
    # -- Stone parameters (curriculum-based)
    stone_width = cfg.stone_params_start.width + difficulty * (
        cfg.stone_params_end.width - cfg.stone_params_start.width
    )
    stone_spacing = cfg.stone_params_start.spacing + difficulty * (
        cfg.stone_params_end.spacing - cfg.stone_params_start.spacing
    )
    stone_height = cfg.stone_params_start.height + difficulty * (
        cfg.stone_params_end.height - cfg.stone_params_start.height
    )
    max_tilt_angle = cfg.stone_params_start.max_tilt_angle + difficulty * (
        cfg.stone_params_end.max_tilt_angle - cfg.stone_params_start.max_tilt_angle
    )
    
    # Initialize list of meshes
    meshes_list = list()
    
    # Compute quantities
    origin = np.asarray((0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0))
    
    # Generate the ground plane
    ground_plane = make_plane(cfg.size, height=-cfg.pit_depth, center_zero=False)
    meshes_list.append(ground_plane)
    
    # Calculate the grid for stepping stones
    grid_size_x = int(cfg.size[0]/stone_spacing) + 1
    grid_size_y = int(cfg.size[1]/stone_spacing) + 1
    
    # Calculate platform boundaries
    platform_x_min = 0.5 * cfg.size[0] - 0.4 * cfg.platform_width
    platform_x_max = 0.5 * cfg.size[0] + 0.4 * cfg.platform_width
    platform_y_min = 0.5 * cfg.size[1] - 0.4 * cfg.platform_width
    platform_y_max = 0.5 * cfg.size[1] + 0.4 * cfg.platform_width
    
    # Generate stepping stones in a grid pattern
    for i in range(grid_size_x):
        for j in range(grid_size_y):
            # Calculate stone position
            x_pos = (i + 0.5) * stone_spacing
            y_pos = (j + 0.5) * stone_spacing
            
            # Skip if the stone would be inside the platform
            if (platform_x_min <= x_pos <= platform_x_max and 
                platform_y_min <= y_pos <= platform_y_max):
                continue
            
            # Skip if the stone would be outside the border
            if (x_pos <= cfg.border_width*0.5 or (cfg.size[0]-cfg.border_width*0.5) <= x_pos) or \
               (y_pos <= cfg.border_width*0.5 or (cfg.size[1]-cfg.border_width*0.5) <= y_pos):
                continue
            
            # Randomly choose stone type from available options
            stone_type = np.random.choice(cfg.stone_types)
            
            # Add random height variation
            height_variation = np.random.uniform(-cfg.height_variation, cfg.height_variation)
            actual_height = stone_height + height_variation
            
            # Create the stone
            if stone_type == "box":
                stone_mesh = make_box(
                    length=stone_width,
                    width=stone_width,
                    height=actual_height,
                    center=(x_pos, y_pos, -actual_height/2),
                    max_yx_angle=max_tilt_angle,
                    degrees=False,  # max_tilt_angle is in radians
                )
            elif stone_type == "cylinder":
                stone_mesh = make_cylinder(
                    radius=stone_width*0.6,
                    height=actual_height,
                    center=(x_pos, y_pos, -actual_height/2),
                    max_yx_angle=max_tilt_angle,
                    degrees=False,  # max_tilt_angle is in radians
                )
            else:
                # Skip unknown stone types
                continue
            
            meshes_list.append(stone_mesh)
    
    def create_random_obstacles(terrain_size, obstacle_num_range, obstacle_size_range, obstacle_height_scale, obstacle_max_tilt_angle, difficulty, platform_x_min, platform_x_max, platform_y_min, platform_y_max):
        """Create random obstacle boxes in the terrain, avoiding platform area."""
        obstacle_meshes = []
        
        # Calculate number of obstacles based on difficulty
        obstacles_num = int(obstacle_num_range[0] + difficulty * (obstacle_num_range[1] - obstacle_num_range[0]))
        
        # Calculate obstacle size range based on difficulty
        obstacle_size = obstacle_size_range[0] + difficulty * (obstacle_size_range[1] - obstacle_size_range[0])
        
        # Maximum attempts to find a valid position for each obstacle
        max_attempts = 50
        
        for _ in range(obstacles_num):
            # Try to find a valid position that doesn't overlap with platform
            valid_position_found = False
            attempts = 0
            
            while not valid_position_found and attempts < max_attempts:
                # Random position within terrain bounds (with margin to avoid edge issues)
                margin = obstacle_size / 2
                obstacle_x = np.random.uniform(margin, terrain_size[0] - margin)
                obstacle_y = np.random.uniform(margin, terrain_size[1] - margin)
                
                # Check if obstacle overlaps with platform
                # We'll use a simple bounding box check
                obstacle_x_min = obstacle_x - obstacle_size/2
                obstacle_x_max = obstacle_x + obstacle_size/2
                obstacle_y_min = obstacle_y - obstacle_size/2
                obstacle_y_max = obstacle_y + obstacle_size/2
                
                # Check for overlap with platform
                x_overlap = (obstacle_x_min < platform_x_max) and (obstacle_x_max > platform_x_min)
                y_overlap = (obstacle_y_min < platform_y_max) and (obstacle_y_max > platform_y_min)
                
                if not (x_overlap and y_overlap):
                    # No overlap with platform, position is valid
                    valid_position_found = True
                else:
                    # Overlap detected, try again
                    attempts += 1
            
            # If we couldn't find a valid position after max attempts, skip this obstacle
            if not valid_position_found:
                continue
                
            # Set z position to 0 (above the pits)
            obstacle_z = 0.0
            
            # Create obstacle cube with only yaw rotation
            obstacle_mesh = make_box(
                length=obstacle_size,
                width=obstacle_size,
                height=obstacle_size * obstacle_height_scale,
                center=(obstacle_x, obstacle_y, obstacle_z),
                max_yx_angle=obstacle_max_tilt_angle,
                degrees=False,
            )
            
            obstacle_meshes.append(obstacle_mesh)
        
        return obstacle_meshes

    # Generate random obstacles in the terrain
    obstacle_meshes = create_random_obstacles(
        cfg.size, 
        cfg.obstacle_num_range,
        cfg.obstacle_size_range,
        cfg.obstacle_height_scale,
        cfg.obstacle_max_tilt_angle,
        difficulty,
        platform_x_min,
        platform_x_max,
        platform_y_min,
        platform_y_max
    )
    meshes_list.extend(obstacle_meshes)

    # generate a platform in the middle
    dim = (cfg.platform_width, cfg.platform_width, cfg.platform_height)
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -0.5 * cfg.platform_height)
    platform = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(platform)

    # generate the border
    border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -cfg.platform_width/2]
    border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
    make_borders = make_border(cfg.size, border_inner_size, cfg.platform_width, border_center)
    # add the border meshes to the list of meshes
    meshes_list += make_borders
    
    return meshes_list, origin


def mesh_platform_bars_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshPlatformBarsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a platform and bars connecting random points on the border.
    
    The terrain has a ground plane with pits and a central platform. Bars are placed
    between random points on opposite sides of the terrain border, creating bridges across the pits.
    
    .. image:: ../../_static/terrains/trimesh/mesh_platform_bars_terrain.jpg
       :width: 40%
       :align: center
    
    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.
    
    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # Initialize list of meshes
    meshes_list = list()
    
    # Compute quantities
    origin = np.asarray((0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0))
    
    def create_bars_connecting_opposite_borders(terrain_size, bar_height, bar_width_range,
                                              bars_num_range, pit_depth, difficulty):
        """Create bar meshes connecting random points on opposite sides of the terrain border."""
        bar_meshes = []
        
        # Calculate number of bars based on difficulty
        bars_num = int(bars_num_range[0] + difficulty * (bars_num_range[1] - bars_num_range[0]))
        
        # Calculate bar width based on difficulty
        bar_width = bar_width_range[1] + difficulty * (bar_width_range[0] - bar_width_range[1])
        
        # Calculate bar center Z coordinate (bars should start from pit bottom and extend upward)
        bar_center_z = -pit_depth + bar_height/2
        
        # Define pairs of opposite borders
        border_pairs = [
            # Bottom and top borders
            [
                lambda: (np.random.uniform(0, terrain_size[0]), 0),  # Bottom
                lambda: (np.random.uniform(0, terrain_size[0]), terrain_size[1])  # Top
            ],
            # Left and right borders
            [
                lambda: (0, np.random.uniform(0, terrain_size[1])),  # Left
                lambda: (terrain_size[0], np.random.uniform(0, terrain_size[1]))  # Right
            ]
        ]
        
        for _ in range(bars_num):
            # Randomly select a pair of opposite borders
            pair_index = np.random.randint(len(border_pairs))
            border_pair = border_pairs[pair_index]
            
            # Get random points on the two opposite borders
            point1 = border_pair[0]()
            point2 = border_pair[1]()
            
            # Calculate the midpoint between the two points
            midpoint_x = (point1[0] + point2[0]) / 2
            midpoint_y = (point1[1] + point2[1]) / 2
            
            # Calculate the distance between the two points (this will be the bar length)
            distance = np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
            
            # Calculate the angle between the two points
            angle = np.arctan2(point2[1] - point1[1], point2[0] - point1[0])
            
            # Create bar using trimesh.creation.box
            bar_dim = (distance, bar_width, bar_height)
            
            # Create transformation matrix with translation and rotation
            translation_matrix = trimesh.transformations.translation_matrix((midpoint_x, midpoint_y, bar_center_z))
            rotation_matrix = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
            transform_matrix = trimesh.transformations.concatenate_matrices(translation_matrix, rotation_matrix)
            
            bar_mesh = trimesh.creation.box(bar_dim, transform=transform_matrix)
            bar_meshes.append(bar_mesh)
        
        return bar_meshes

    # Create bars connecting opposite borders
    bars_meshes = create_bars_connecting_opposite_borders(
        cfg.size, 
        cfg.bar_height,
        cfg.bar_width_range,
        cfg.bars_num_range,
        cfg.pit_depth,
        difficulty,
    )
    
    # Add bars to the meshes list
    meshes_list.extend(bars_meshes)
    
    # Create the central platform
    platform_height = cfg.platform_height if cfg.platform_height >= 0.0 else cfg.bar_height
    platform_dim = (cfg.platform_width, cfg.platform_width, platform_height)
    platform_pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -cfg.pit_depth + platform_height/2)
    platform_mesh = trimesh.creation.box(platform_dim, trimesh.transformations.translation_matrix(platform_pos))
    meshes_list.append(platform_mesh)

    # Create the ground plane at pit bottom
    ground_plane = make_plane(cfg.size, height=-cfg.pit_depth, center_zero=False)
    meshes_list.append(ground_plane)
    
    # Generate the border
    border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -cfg.platform_width/2]
    border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
    make_borders = make_border(cfg.size, border_inner_size, cfg.platform_width, border_center)
    # Add the border meshes to the list of meshes
    meshes_list.extend(make_borders)

    return meshes_list, origin

