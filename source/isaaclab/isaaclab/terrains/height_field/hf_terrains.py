# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions to generate height fields for different terrains."""

from __future__ import annotations

import numpy as np
import scipy.interpolate as interpolate
from typing import TYPE_CHECKING
from PIL import Image

from .utils import height_field_to_mesh, height_field_to_mesh_center

if TYPE_CHECKING:
    from . import hf_terrains_cfg


@height_field_to_mesh
def random_uniform_terrain(difficulty: float, cfg: hf_terrains_cfg.HfRandomUniformTerrainCfg) -> np.ndarray:
    """Generate a terrain with height sampled uniformly from a specified range.

    .. image:: ../../_static/terrains/height_field/random_uniform_terrain.jpg
       :width: 40%
       :align: center

    Note:
        The :obj:`difficulty` parameter is ignored for this terrain.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.

    Raises:
        ValueError: When the downsampled scale is smaller than the horizontal scale.
    """
    # check parameters
    # -- horizontal scale
    if cfg.downsampled_scale is None:
        cfg.downsampled_scale = cfg.horizontal_scale
    elif cfg.downsampled_scale < cfg.horizontal_scale:
        raise ValueError(
            "Downsampled scale must be larger than or equal to the horizontal scale:"
            f" {cfg.downsampled_scale} < {cfg.horizontal_scale}."
        )

    # switch parameters to discrete units
    # -- horizontal scale
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- downsampled scale
    width_downsampled = int(cfg.size[0] / cfg.downsampled_scale)
    length_downsampled = int(cfg.size[1] / cfg.downsampled_scale)
    # -- height
    height_min = int(cfg.noise_range[0] / cfg.vertical_scale)
    height_max = int(cfg.noise_range[1] / cfg.vertical_scale)
    height_step = int(cfg.noise_step / cfg.vertical_scale)

    # create range of heights possible
    height_range = np.arange(height_min, height_max + height_step, height_step)
    # sample heights randomly from the range along a grid
    height_field_downsampled = np.random.choice(height_range, size=(width_downsampled, length_downsampled))
    # create interpolation function for the sampled heights
    x = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_downsampled)
    y = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_downsampled)
    func = interpolate.RectBivariateSpline(x, y, height_field_downsampled)

    # interpolate the sampled heights to obtain the height field
    x_upsampled = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_pixels)
    y_upsampled = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_pixels)
    z_upsampled = func(x_upsampled, y_upsampled)
    # round off the interpolated heights to the nearest vertical step
    return np.rint(z_upsampled).astype(np.int16)


@height_field_to_mesh
def pyramid_sloped_terrain(difficulty: float, cfg: hf_terrains_cfg.HfPyramidSlopedTerrainCfg) -> np.ndarray:
    """Generate a terrain with a truncated pyramid structure.

    The terrain is a pyramid-shaped sloped surface with a slope of :obj:`slope` that trims into a flat platform
    at the center. The slope is defined as the ratio of the height change along the x axis to the width along the
    x axis. For example, a slope of 1.0 means that the height changes by 1 unit for every 1 unit of width.

    If the :obj:`cfg.inverted` flag is set to :obj:`True`, the terrain is inverted such that
    the platform is at the bottom.

    .. image:: ../../_static/terrains/height_field/pyramid_sloped_terrain.jpg
       :width: 40%

    .. image:: ../../_static/terrains/height_field/inverted_pyramid_sloped_terrain.jpg
       :width: 40%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """
    # resolve terrain configuration
    if cfg.inverted:
        slope = -cfg.slope_range[0] - difficulty * (cfg.slope_range[1] - cfg.slope_range[0])
    else:
        slope = cfg.slope_range[0] + difficulty * (cfg.slope_range[1] - cfg.slope_range[0])

    # switch parameters to discrete units
    # -- horizontal scale
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- height
    # we want the height to be 1/2 of the width since the terrain is a pyramid
    height_max = int(slope * cfg.size[0] / 2 / cfg.vertical_scale)
    # -- center of the terrain
    center_x = int(width_pixels / 2)
    center_y = int(length_pixels / 2)

    # create a meshgrid of the terrain
    x = np.arange(0, width_pixels)
    y = np.arange(0, length_pixels)
    xx, yy = np.meshgrid(x, y, sparse=True)
    # offset the meshgrid to the center of the terrain
    xx = (center_x - np.abs(center_x - xx)) / center_x
    yy = (center_y - np.abs(center_y - yy)) / center_y
    # reshape the meshgrid to be 2D
    xx = xx.reshape(width_pixels, 1)
    yy = yy.reshape(1, length_pixels)
    # create a sloped surface
    hf_raw = np.zeros((width_pixels, length_pixels))
    hf_raw = height_max * xx * yy

    # create a flat platform at the center of the terrain
    platform_width = int(cfg.platform_width / cfg.horizontal_scale / 2)
    # get the height of the platform at the corner of the platform
    x_pf = width_pixels // 2 - platform_width
    y_pf = length_pixels // 2 - platform_width
    z_pf = hf_raw[x_pf, y_pf]
    hf_raw = np.clip(hf_raw, min(0, z_pf), max(0, z_pf))

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def pyramid_stairs_terrain(difficulty: float, cfg: hf_terrains_cfg.HfPyramidStairsTerrainCfg) -> np.ndarray:
    """Generate a terrain with a pyramid stair pattern.

    The terrain is a pyramid stair pattern which trims to a flat platform at the center of the terrain.

    If the :obj:`cfg.inverted` flag is set to :obj:`True`, the terrain is inverted such that
    the platform is at the bottom.

    .. image:: ../../_static/terrains/height_field/pyramid_stairs_terrain.jpg
       :width: 40%

    .. image:: ../../_static/terrains/height_field/inverted_pyramid_stairs_terrain.jpg
       :width: 40%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """
    # resolve terrain configuration
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    if cfg.inverted:
        step_height *= -1
    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- stairs
    step_width = int(cfg.step_width / cfg.horizontal_scale)
    step_height = int(step_height / cfg.vertical_scale)
    # -- platform
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)

    # create a terrain with a flat platform at the center
    hf_raw = np.zeros((width_pixels, length_pixels))
    # add the steps
    current_step_height = 0
    start_x, start_y = 0, 0
    stop_x, stop_y = width_pixels, length_pixels
    while (stop_x - start_x) > platform_width and (stop_y - start_y) > platform_width:
        # increment position
        # -- x
        start_x += step_width
        stop_x -= step_width
        # -- y
        start_y += step_width
        stop_y -= step_width
        # increment height
        current_step_height += step_height
        # add the step
        hf_raw[start_x:stop_x, start_y:stop_y] = current_step_height

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def discrete_obstacles_terrain(difficulty: float, cfg: hf_terrains_cfg.HfDiscreteObstaclesTerrainCfg) -> np.ndarray:
    """Generate a terrain with randomly generated obstacles as pillars with positive and negative heights.

    The terrain is a flat platform at the center of the terrain with randomly generated obstacles as pillars
    with positive and negative height. The obstacles are randomly generated cuboids with a random width and
    height. They are placed randomly on the terrain with a minimum distance of :obj:`cfg.platform_width`
    from the center of the terrain.

    .. image:: ../../_static/terrains/height_field/discrete_obstacles_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """
    # resolve terrain configuration
    obs_height = cfg.obstacle_height_range[0] + difficulty * (
        cfg.obstacle_height_range[1] - cfg.obstacle_height_range[0]
    )

    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- obstacles
    obs_height = int(obs_height / cfg.vertical_scale)
    obs_width_min = int(cfg.obstacle_width_range[0] / cfg.horizontal_scale)
    obs_width_max = int(cfg.obstacle_width_range[1] / cfg.horizontal_scale)
    # -- center of the terrain
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)

    # create discrete ranges for the obstacles
    # -- shape
    obs_width_range = np.arange(obs_width_min, obs_width_max, 4)
    obs_length_range = np.arange(obs_width_min, obs_width_max, 4)
    # -- position
    obs_x_range = np.arange(0, width_pixels, 4)
    obs_y_range = np.arange(0, length_pixels, 4)

    # create a terrain with a flat platform at the center
    hf_raw = np.zeros((width_pixels, length_pixels))
    # generate the obstacles
    for _ in range(cfg.num_obstacles):
        # sample size
        if cfg.obstacle_height_mode == "choice":
            height = np.random.choice([-obs_height, -obs_height // 2, obs_height // 2, obs_height])
        elif cfg.obstacle_height_mode == "fixed":
            height = obs_height
        else:
            raise ValueError(f"Unknown obstacle height mode '{cfg.obstacle_height_mode}'. Must be 'choice' or 'fixed'.")
        width = int(np.random.choice(obs_width_range))
        length = int(np.random.choice(obs_length_range))
        # sample position
        x_start = int(np.random.choice(obs_x_range))
        y_start = int(np.random.choice(obs_y_range))
        # clip start position to the terrain
        if x_start + width > width_pixels:
            x_start = width_pixels - width
        if y_start + length > length_pixels:
            y_start = length_pixels - length
        # add to terrain
        hf_raw[x_start : x_start + width, y_start : y_start + length] = height
    # clip the terrain to the platform
    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0
    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def wave_terrain(difficulty: float, cfg: hf_terrains_cfg.HfWaveTerrainCfg) -> np.ndarray:
    r"""Generate a terrain with a wave pattern.

    The terrain is a flat platform at the center of the terrain with a wave pattern. The wave pattern
    is generated by adding sinusoidal waves based on the number of waves and the amplitude of the waves.

    The height of the terrain at a point :math:`(x, y)` is given by:

    .. math::

        h(x, y) =  A \left(\sin\left(\frac{2 \pi x}{\lambda}\right) + \cos\left(\frac{2 \pi y}{\lambda}\right) \right)

    where :math:`A` is the amplitude of the waves, :math:`\lambda` is the wavelength of the waves.

    .. image:: ../../_static/terrains/height_field/wave_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.

    Raises:
        ValueError: When the number of waves is non-positive.
    """
    # check number of waves
    if cfg.num_waves < 0:
        raise ValueError(f"Number of waves must be a positive integer. Got: {cfg.num_waves}.")

    # resolve terrain configuration
    amplitude = cfg.amplitude_range[0] + difficulty * (cfg.amplitude_range[1] - cfg.amplitude_range[0])
    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    amplitude_pixels = int(0.5 * amplitude / cfg.vertical_scale)

    # compute the wave number: nu = 2 * pi / lambda
    wave_length = length_pixels / cfg.num_waves
    wave_number = 2 * np.pi / wave_length
    # create meshgrid for the terrain
    x = np.arange(0, width_pixels)
    y = np.arange(0, length_pixels)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = xx.reshape(width_pixels, 1)
    yy = yy.reshape(1, length_pixels)

    # create a terrain with a flat platform at the center
    hf_raw = np.zeros((width_pixels, length_pixels))
    # add the waves
    hf_raw += amplitude_pixels * (np.cos(yy * wave_number) + np.sin(xx * wave_number))
    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def stepping_stones_terrain(difficulty: float, cfg: hf_terrains_cfg.HfSteppingStonesTerrainCfg) -> np.ndarray:
    """Generate a terrain with a stepping stones pattern.

    The terrain is a stepping stones pattern which trims to a flat platform at the center of the terrain.

    .. image:: ../../_static/terrains/height_field/stepping_stones_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """
    # resolve terrain configuration
    stone_width = cfg.stone_width_range[1] - difficulty * (cfg.stone_width_range[1] - cfg.stone_width_range[0])
    stone_distance = cfg.stone_distance_range[0] + difficulty * (
        cfg.stone_distance_range[1] - cfg.stone_distance_range[0]
    )

    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- stones
    stone_distance = int(stone_distance / cfg.horizontal_scale)
    stone_width = int(stone_width / cfg.horizontal_scale)
    stone_height_max = int(cfg.stone_height_max / cfg.vertical_scale)
    # -- holes
    holes_depth = int(cfg.holes_depth / cfg.vertical_scale)
    # -- platform
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)
    # create range of heights
    stone_height_range = np.arange(-stone_height_max - 1, stone_height_max, step=1)

    # create a terrain with a flat platform at the center
    hf_raw = np.full((width_pixels, length_pixels), holes_depth)
    # add the stones
    start_x, start_y = 0, 0
    # -- if the terrain is longer than it is wide then fill the terrain column by column
    if length_pixels >= width_pixels:
        while start_y < length_pixels:
            # ensure that stone stops along y-axis
            stop_y = min(length_pixels, start_y + stone_width)
            # randomly sample x-position
            start_x = np.random.randint(0, stone_width)
            stop_x = max(0, start_x - stone_distance)
            # fill first stone
            hf_raw[0:stop_x, start_y:stop_y] = np.random.choice(stone_height_range)
            # fill row with stones
            while start_x < width_pixels:
                stop_x = min(width_pixels, start_x + stone_width)
                hf_raw[start_x:stop_x, start_y:stop_y] = np.random.choice(stone_height_range)
                start_x += stone_width + stone_distance
            # update y-position
            start_y += stone_width + stone_distance
    elif width_pixels > length_pixels:
        while start_x < width_pixels:
            # ensure that stone stops along x-axis
            stop_x = min(width_pixels, start_x + stone_width)
            # randomly sample y-position
            start_y = np.random.randint(0, stone_width)
            stop_y = max(0, start_y - stone_distance)
            # fill first stone
            hf_raw[start_x:stop_x, 0:stop_y] = np.random.choice(stone_height_range)
            # fill column with stones
            while start_y < length_pixels:
                stop_y = min(length_pixels, start_y + stone_width)
                hf_raw[start_x:stop_x, start_y:stop_y] = np.random.choice(stone_height_range)
                start_y += stone_width + stone_distance
            # update x-position
            start_x += stone_width + stone_distance
    # add the platform in the center
    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0
    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def stepping_slope_stones_terrain(difficulty: float, cfg: hf_terrains_cfg.HfSteppingSlopeStonesTerrainCfg) -> np.ndarray:
    """Generate a terrain with a stepping stones pattern with sloped tops."""
    
    # 解析地形配置
    stone_width = cfg.stone_width_range[1] - difficulty * (cfg.stone_width_range[1] - cfg.stone_width_range[0])
    stone_distance = cfg.stone_distance_range[0] + difficulty * (
        cfg.stone_distance_range[1] - cfg.stone_distance_range[0]
    )

    # 单位转换
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    stone_distance = int(stone_distance / cfg.horizontal_scale)
    stone_width = int(stone_width / cfg.horizontal_scale)
    stone_height_max = int(cfg.stone_height_max / cfg.vertical_scale)
    holes_depth = int(cfg.holes_depth / cfg.vertical_scale)
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)

    # 创建高度范围
    stone_height_range = np.arange(-stone_height_max - 1, stone_height_max, step=1)

    # 初始化地形
    hf_raw = np.full((width_pixels, length_pixels), holes_depth)
    
    # 首先添加中心平台
    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0
    
    # 辅助函数：检查位置是否与平台重叠
    def overlaps_with_platform(start_x, start_y, stone_width):
        """检查位置是否与平台重叠"""
        stop_x = min(width_pixels, start_x + stone_width)
        stop_y = min(length_pixels, start_y + stone_width)
        
        # 计算石头与平台的重叠区域
        overlap_x1 = max(start_x, x1)
        overlap_x2 = min(stop_x, x2)
        overlap_y1 = max(start_y, y1)
        overlap_y2 = min(stop_y, y2)
        
        # 如果有重叠，则返回True
        return overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2
    
    # 辅助函数：创建倾斜的石头
    def create_sloped_stone(height_field, start_x, start_y, stone_width):
        """在指定位置创建具有斜面的石头"""
        stop_x = min(width_pixels, start_x + stone_width)
        stop_y = min(length_pixels, start_y + stone_width)
        
        # 随机选择基础高度
        base_height = np.random.choice(stone_height_range)
        
        # 随机选择斜面方向：0=x方向, 1=y方向, 2=对角线方向
        slope_direction = np.random.randint(0, 3)
        
        # 随机选择坡度值（与难度脱钩）
        stone_slope = np.random.uniform(cfg.stone_slope_range[0], cfg.stone_slope_range[1])
        
        # 随机选择斜面方向：0=x方向, 1=y方向, 2=对角线方向
        if slope_direction == 0:  # x方向斜面
            for i in range(start_x, stop_x):
                # 计算从起始位置的相对距离（以cfg.horizontal_scale为单位）
                relative_distance = (i - start_x) * cfg.horizontal_scale
                # 计算高度变化
                height_offset = stone_slope * relative_distance / cfg.vertical_scale
                height_field[i, start_y:stop_y] = base_height + height_offset
                
        elif slope_direction == 1:  # y方向斜面
            for j in range(start_y, stop_y):
                # 计算从起始位置的相对距离（以cfg.horizontal_scale为单位）
                relative_distance = (j - start_y) * cfg.horizontal_scale
                # 计算高度变化
                height_offset = stone_slope * relative_distance / cfg.vertical_scale
                height_field[start_x:stop_x, j] = base_height + height_offset
                
        else:  # 对角线斜面
            for i in range(start_x, stop_x):
                for j in range(start_y, stop_y):
                    # 计算对角线方向的相对距离
                    distance_x = (i - start_x) * cfg.horizontal_scale
                    distance_y = (j - start_y) * cfg.horizontal_scale
                    diagonal_distance = (distance_x + distance_y) / 2  # 平均距离
                    # 计算高度变化
                    height_offset = stone_slope * diagonal_distance / cfg.vertical_scale
                    height_field[i, j] = base_height + height_offset
    
    # 辅助函数：创建平整的石头（与平台齐平）
    def create_flat_stone(height_field, start_x, start_y, stone_width):
        """在指定位置创建与平台齐平的平整石头"""
        stop_x = min(width_pixels, start_x + stone_width)
        stop_y = min(length_pixels, start_y + stone_width)
        
        # 将整个石头区域设置为平台高度（0）
        height_field[start_x:stop_x, start_y:stop_y] = 0
    
    # 添加石头 - 修改后的逻辑，与平台重叠的石头生成平整版本
    start_x, start_y = 0, 0
    
    if length_pixels >= width_pixels:
        while start_y < length_pixels:
            stop_y = min(length_pixels, start_y + stone_width)
            start_x = np.random.randint(0, stone_width)
            stop_x = max(0, start_x - stone_distance)
            
            # 创建第一个石头
            if stop_x > 0:
                # 检查是否与平台重叠
                if overlaps_with_platform(0, start_y, stop_x):
                    create_flat_stone(hf_raw, 0, start_y, stop_x)
                else:
                    create_sloped_stone(hf_raw, 0, start_y, stop_x)
            
            # 填充行中的其他石头
            while start_x < width_pixels:
                stop_x = min(width_pixels, start_x + stone_width)
                if stop_x > start_x:
                    # 检查是否与平台重叠
                    if overlaps_with_platform(start_x, start_y, stone_width):
                        create_flat_stone(hf_raw, start_x, start_y, stone_width)
                    else:
                        create_sloped_stone(hf_raw, start_x, start_y, stone_width)
                start_x += stone_width + stone_distance
            
            start_y += stone_width + stone_distance
            
    elif width_pixels > length_pixels:
        while start_x < width_pixels:
            stop_x = min(width_pixels, start_x + stone_width)
            start_y = np.random.randint(0, stone_width)
            stop_y = max(0, start_y - stone_distance)
            
            # 创建第一个石头
            if stop_y > 0:
                # 检查是否与平台重叠
                if overlaps_with_platform(start_x, 0, stone_width):
                    create_flat_stone(hf_raw, start_x, 0, stop_y)
                else:
                    create_sloped_stone(hf_raw, start_x, 0, stop_y)
            
            # 填充列中的其他石头
            while start_y < length_pixels:
                stop_y = min(length_pixels, start_y + stone_width)
                if stop_y > start_y:
                    # 检查是否与平台重叠
                    if overlaps_with_platform(start_x, start_y, stone_width):
                        create_flat_stone(hf_raw, start_x, start_y, stone_width)
                    else:
                        create_sloped_stone(hf_raw, start_x, start_y, stone_width)
                start_y += stone_width + stone_distance
            
            start_x += stone_width + stone_distance

    return np.rint(hf_raw).astype(np.int16)


# ====================================================================================================
# Image-based terrain generation functions
# ====================================================================================================


def _load_image(image_path: str, target_size: tuple[int, int] | None = None,
                interpolation: str = "bilinear", save_resized: bool = False) -> np.ndarray:
    """Load an image with optional resizing and saving.

    Args:
        image_path: Path to the image file.
        target_size: Optional target size as (width, height) in pixels.
        interpolation: Interpolation method when resizing.
        save_resized: If True, save the resized image to a file.

    Returns:
        The image as a numpy array with dimensions (height, width, channels).
    """
    img = Image.open(image_path)
    
    # Resize to target dimensions if specified
    if target_size is not None:
        img = img.resize(target_size, Image.Resampling[interpolation.upper()])
        
        # Save resized image if requested
        if save_resized:
            # Generate output path by adding '_resize' before the extension
            import os
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_resize{ext}"
            img.save(output_path)
            print(f"Resized image saved to: {output_path}")
    
    img_array = np.array(img)
    return img_array


def _extract_height_from_image(image: np.ndarray, color_mode: str, hsv_zero_point: float = 180.0, grayscale_zero_point: float = 127.5) -> np.ndarray:
    """Extract height information from an image based on the specified color mode.

    Args:
        image: Input image as numpy array.
        color_mode: Color mode for height extraction ("grayscale" or "hsv").
        hsv_zero_point: Hue value corresponding to zero height (0-360). Defaults to 180.0.
        grayscale_zero_point: Grayscale value corresponding to zero height (0-255). Defaults to 127.5.

    Returns:
        Height field as 2D numpy array with raw values.
        
        For both modes: Values range from 0.0 to 1.0 where 0.5 corresponds to zero height.
        - Values > 0.5: Elevated terrain (convex)
        - Values < 0.5: Depressed terrain (concave)
        - Value = 0.5: Zero height (baseline)
    """
    if color_mode == "grayscale":
        # Convert to grayscale using luminance formula
        if len(image.shape) == 3:
            height_field = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            height_field = image.astype(np.float32)
        
        # Map grayscale to height based on zero_point
        # value > grayscale_zero_point: elevated (convex)
        # value < grayscale_zero_point: depressed (concave)
        # value == grayscale_zero_point: zero height
        height_field = height_field - grayscale_zero_point
    
    elif color_mode == "hsv":
        # Use hue value from HSV color space with zero-point mapping
        from skimage.color import rgb2hsv
        
        # Convert RGBA to RGB if needed (take only first 3 channels)
        if image.shape[2] == 4:
            image_rgb = image[:, :, :3]
        else:
            image_rgb = image
        
        hsv_image = rgb2hsv(image_rgb.astype(np.float32) / 255.0)
        hue = hsv_image[:, :, 0] * 360.0  # Hue in degrees 0-360
        
        # Map hue to height based on zero_point
        # hue > hsv_zero_point: elevated (convex)
        # hue < hsv_zero_point: depressed (concave)
        # hue == hsv_zero_point: zero height
        height_field = hue - hsv_zero_point
            
    else:
        raise ValueError(f"Unknown color mode: {color_mode}")
    
    return height_field


def _process_single_image(cfg: "hf_terrains_cfg.HfImageBasedTerrainCfg", difficulty: float = 1.0) -> np.ndarray:
    """Process a single image to generate height field.

    The image is processed to match the terrain dimensions, then converted to discretized heights.
    Processing order: load/resize → normalize → scale → discretize to maintain precision.

    Args:
        cfg: Image-based terrain configuration (may be modified by decorator).

    Returns:
        Height field as 2D numpy array with discretized heights (int16).

    Raises:
        ValueError: If image dimensions exceed the maximum allowed by terrain size and horizontal scale.
    """
    # Calculate target pixel dimensions based on current cfg.size (modified by decorator)
    target_width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    target_length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    # Load image with optional resizing
    if cfg.resize_to_target:
        # Resize image to match target dimensions
        image = _load_image(cfg.image_path,
                            target_size=(target_width_pixels, target_length_pixels),
                            interpolation=cfg.resize_interpolation,
                            save_resized=cfg.save_resized_image)
    else:
        # Load image with original dimensions
        image = _load_image(cfg.image_path)

    # Get actual image dimensions (height, width)
    img_height, img_width = image.shape[:2]

    # Validate image dimensions (only needed when not resizing)
    if not cfg.resize_to_target:
        if img_width > target_width_pixels:
            raise ValueError(
                f"Image width ({img_width} pixels) exceeds maximum allowed width "
                f"({target_width_pixels} pixels) for terrain size {cfg.size[0]}m "
                f"with horizontal scale {cfg.horizontal_scale}m. "
                f"Image path: {cfg.image_path}"
            )
        if img_height > target_length_pixels:
            raise ValueError(
                f"Image height ({img_height} pixels) exceeds maximum allowed height "
                f"({target_length_pixels} pixels) for terrain size {cfg.size[1]}m "
                f"with horizontal scale {cfg.horizontal_scale}m. "
                f"Image path: {cfg.image_path}"
            )

    # Extract height from image (raw values)
    height_field = _extract_height_from_image(image, cfg.color_mode, cfg.hsv_zero_point, cfg.grayscale_zero_point)
    # Apply height scaling (multiply by scale factor)
    height_field = height_field * cfg.height_scale * difficulty

    # Center height field if needed (only when not resizing)
    if not cfg.resize_to_target and (img_width != target_width_pixels or img_height != target_length_pixels):
        # Create target array filled with zeros
        target_height_field = np.zeros((target_length_pixels, target_width_pixels), dtype=np.float32)

        # Calculate offset to center the height field
        offset_x = (target_width_pixels - img_width) // 2
        offset_y = (target_length_pixels - img_height) // 2

        # Place height field in center
        target_height_field[offset_y:offset_y + img_height, offset_x:offset_x + img_width] = height_field
        height_field = target_height_field

    return height_field


def _process_multi_layer(cfg: "hf_terrains_cfg.HfImageBasedTerrainCfg", difficulty: float = 1.0) -> np.ndarray:
    """Process multiple image layers with blending.

    All images are processed to match the terrain dimensions, then converted to discretized heights.
    Processing order: load/resize → normalize → blend → scale → discretize to maintain precision.

    Args:
        cfg: Image-based terrain configuration with layer_configs (may be modified by decorator).

    Returns:
        Combined height field as 2D numpy array with discretized heights (int16).

    Raises:
        ValueError: If any image dimensions exceed the maximum allowed or if images have different dimensions.
    """
    # Calculate target pixel dimensions based on current cfg.size (modified by decorator)
    target_width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    target_length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    
    # Load images with optional resizing
    if cfg.resize_to_target:
        # Resize all images to match target dimensions
        first_image = _load_image(cfg.layer_configs[0].image_path,
                                target_size=(target_width_pixels, target_length_pixels),
                                interpolation=cfg.resize_interpolation,
                                save_resized=cfg.save_resized_image)
        img_height, img_width = first_image.shape[:2]
    else:
        # Load images with original dimensions
        first_image = _load_image(cfg.layer_configs[0].image_path)
        img_height, img_width = first_image.shape[:2]
        
        # Validate first image dimensions
        if img_width > target_width_pixels:
            raise ValueError(
                f"Image width ({img_width} pixels) exceeds maximum allowed width "
                f"({target_width_pixels} pixels) for terrain size {cfg.size[0]}m "
                f"with horizontal scale {cfg.horizontal_scale}m. "
                f"Image path: {cfg.layer_configs[0].image_path}"
            )
        
        if img_height > target_length_pixels:
            raise ValueError(
                f"Image height ({img_height} pixels) exceeds maximum allowed height "
                f"({target_length_pixels} pixels) for terrain size {cfg.size[1]}m "
                f"with horizontal scale {cfg.horizontal_scale}m. "
                f"Image path: {cfg.layer_configs[0].image_path}"
            )
    
    # Initialize combined height field with image dimensions
    combined_height = np.zeros((img_height, img_width), dtype=np.float32)
    
    # Process each layer
    for i, layer_cfg in enumerate(cfg.layer_configs):
        # Load image
        if cfg.resize_to_target:
            image = _load_image(layer_cfg.image_path,
                              target_size=(target_width_pixels, target_length_pixels),
                              interpolation=cfg.resize_interpolation,
                              save_resized=cfg.save_resized_image)
        else:
            image = _load_image(layer_cfg.image_path)
            
            # Verify dimensions match first image
            if image.shape[:2] != (img_height, img_width):
                raise ValueError(
                    f"All images in multi-layer configuration must have the same dimensions. "
                    f"First image ({cfg.layer_configs[0].image_path}): ({img_height}, {img_width}), "
                    f"Layer {i+1} image ({layer_cfg.image_path}): {image.shape[:2]}"
                )
        
        # Extract height from image (raw values)
        layer_height = _extract_height_from_image(image, layer_cfg.color_mode, cfg.hsv_zero_point, cfg.grayscale_zero_point)
        # Apply layer-specific scaling
        layer_height = layer_height * layer_cfg.height_scale * difficulty
        
        # Apply layer-specific smoothing if configured
        if layer_cfg.smoothing_sigma > 0:
            layer_height = _smooth_layer(height_field=layer_height, smoothing_sigma=layer_cfg.smoothing_sigma)
        
        # Blend with combined height field
        if layer_cfg.blend_mode == "add":
            combined_height += layer_height
        elif layer_cfg.blend_mode == "multiply":
            combined_height *= (1.0 + layer_height)
        elif layer_cfg.blend_mode == "max":
            combined_height = np.maximum(combined_height, layer_height)
        elif layer_cfg.blend_mode == "min":
            combined_height = np.minimum(combined_height, layer_height)
        else:
            raise ValueError(f"Unknown blend mode: {layer_cfg.blend_mode}")
    
    # Center height field if needed (only when not resizing)
    if not cfg.resize_to_target and (img_width != target_width_pixels or img_height != target_length_pixels):
        # Create target array filled with zeros
        target_height_field = np.zeros((target_length_pixels, target_width_pixels), dtype=np.float32)
        
        # Calculate offset to center the height field
        offset_x = (target_width_pixels - img_width) // 2
        offset_y = (target_length_pixels - img_height) // 2
        
        # Place height field in center
        target_height_field[offset_y:offset_y + img_height, offset_x:offset_x + img_width] = combined_height
        combined_height = target_height_field
    
    return combined_height


def _smooth_layer(height_field: np.ndarray, smoothing_sigma: float) -> np.ndarray:
    """Apply Gaussian smoothing to a height field layer.

    Args:
        height_field: Input height field.
        smoothing_sigma: Smoothing sigma (radius) in pixels.

    Returns:
        Smoothed height field.
    """
    from scipy.ndimage import gaussian_filter
    
    # Apply Gaussian smoothing
    smoothed = gaussian_filter(height_field, sigma=smoothing_sigma)
    
    return smoothed


def _add_platform(height_field: np.ndarray, cfg: "hf_terrains_cfg.HfImageBasedTerrainCfg") -> np.ndarray:
    """Add a flat platform at the center of the terrain.

    Args:
        height_field: Input height field.
        cfg: Terrain configuration.

    Returns:
        Height field with platform added.
    """
    if cfg.platform_width <= 0:
        return height_field
    
    # Calculate platform dimensions in pixels
    width_pixels, length_pixels = height_field.shape
    platform_width_pixels = int(cfg.platform_width / cfg.horizontal_scale)
    
    # Calculate platform boundaries
    x1 = (width_pixels - platform_width_pixels) // 2
    x2 = (width_pixels + platform_width_pixels) // 2
    y1 = (length_pixels - platform_width_pixels) // 2
    y2 = (length_pixels + platform_width_pixels) // 2
    
    # Get height at platform edges for smooth transition
    platform_height = np.mean([
        height_field[x1, y1], height_field[x1, y2],
        height_field[x2, y1], height_field[x2, y2]
    ])
    
    # Set platform area to flat height
    height_field[x1:x2, y1:y2] = platform_height
    
    return height_field


def _remove_geometric_artifacts(height_field: np.ndarray, height_threshold: float = 0.5) -> np.ndarray:
    """
    精确移除由多图层叠加产生的“零厚度墙”和几何伪影。
    使用8邻域逻辑进行扫描。
    """
    repaired_hf = height_field.copy()
    rows, cols = height_field.shape
    
    # 定义8-邻域偏移量 (上, 下, 左, 右, 左上, 右上, 左下, 右下)
    # 这样可以检测所有方向的连通性
    neighbor_offsets = [
        (-1, 0), (1, 0), (0, -1), (0, 1), # 4-邻域
        (-1, -1), (-1, 1), (1, -1), (1, 1) # 对角线
    ]
    
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            center_val = height_field[i, j]
            
            # 1. 提取8-邻域的高度值
            neighbor_vals = []
            for di, dj in neighbor_offsets:
                neighbor_vals.append(height_field[i + di, j + dj])
            
            neighbor_vals = np.array(neighbor_vals)
            
            # 2. 计算与8-邻域的最大高度差 (梯度检测)
            # 使用8邻域能更敏锐地捕捉到对角线方向的突变
            max_diff = np.max(np.abs(neighbor_vals - center_val))
            
            # 如果高度变化平缓，说明是平地或缓坡，直接跳过
            if max_diff < height_threshold:
                continue
            
            # 统计有多少个点与中心点高度相似
            similar_count = np.sum(np.abs(neighbor_vals - center_val) < (height_threshold / 2))
            if similar_count <= 2: 
                repaired_hf[i, j] = np.mean(neighbor_vals)
                
    return repaired_hf


@height_field_to_mesh_center
def image_based_terrain(difficulty: float, cfg: "hf_terrains_cfg.HfImageBasedTerrainCfg") -> np.ndarray:
    """Generate a terrain based on image files.

    This function generates terrains from images where pixel colors are mapped to terrain heights.
    Supports both single-image and multi-layer blending modes.

    The :obj:`difficulty` parameter scales the overall height of the terrain.

    .. image:: ../../_static/terrains/height_field/image_based_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. This value between 0 and 1 scales the terrain height.
        cfg: The configuration for the image-based terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.

    Raises:
        ValueError: When neither image_path nor layer_configs is provided.
    """
    # Determine mode: single image or multi-layer
    if cfg.layer_configs is not None:
        # Multi-layer blending mode
        height_field = _process_multi_layer(cfg, difficulty)
    elif cfg.image_path is not None:
        # Single image mode
        height_field = _process_single_image(cfg, difficulty)
    else:
        raise ValueError("Either 'image_path' or 'layer_configs' must be provided in the configuration.")
    
    # Invert if requested
    if cfg.invert_height:
        height_field = height_field * (-1.0)
    
    # Add center platform if requested (operates on discretized heights)
    if cfg.platform_width > 0:
        height_field = _add_platform(height_field, cfg)
        
    # Remove thin "wall"
    height_field = _remove_geometric_artifacts(height_field, height_threshold=0.1)
    
    # Apply terrain smoothing if requested (operates on discretized heights)
    if cfg.terrain_smoothing_sigma > 0:
        height_field = _smooth_layer(height_field, cfg.terrain_smoothing_sigma)

    # Convert to discretized heights (divide by vertical_scale and round)
    height_field = np.rint(height_field / cfg.vertical_scale).astype(np.int16)
    
    return height_field

