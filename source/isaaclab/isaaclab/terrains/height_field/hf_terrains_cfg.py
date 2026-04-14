# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass

from ..sub_terrain_cfg import SubTerrainBaseCfg
from . import hf_terrains

from typing import Literal

@configclass
class HfTerrainBaseCfg(SubTerrainBaseCfg):
    """The base configuration for height field terrains."""

    border_width: float = 0.0
    """The width of the border/padding around the terrain (in m). Defaults to 0.0.

    The border width is subtracted from the :obj:`size` of the terrain. If non-zero, it must be
    greater than or equal to the :obj:`horizontal scale`.
    """

    horizontal_scale: float = 0.1
    """The discretization of the terrain along the x and y axes (in m). Defaults to 0.1."""

    vertical_scale: float = 0.005
    """The discretization of the terrain along the z axis (in m). Defaults to 0.005."""

    slope_threshold: float | None = None
    """The slope threshold above which surfaces are made vertical. Defaults to None,
    in which case no correction is applied."""


"""
Different height field terrain configurations.
"""


@configclass
class HfRandomUniformTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a random uniform height field terrain."""

    function = hf_terrains.random_uniform_terrain

    noise_range: tuple[float, float] = MISSING
    """The minimum and maximum height noise (i.e. along z) of the terrain (in m)."""

    noise_step: float = MISSING
    """The minimum height (in m) change between two points."""

    downsampled_scale: float | None = None
    """The distance between two randomly sampled points on the terrain. Defaults to None,
    in which case the :obj:`horizontal scale` is used.

    The heights are sampled at this resolution and interpolation is performed for intermediate points.
    This must be larger than or equal to the :obj:`horizontal scale`.
    """


@configclass
class HfPyramidSlopedTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a pyramid sloped height field terrain."""

    function = hf_terrains.pyramid_sloped_terrain

    slope_range: tuple[float, float] = MISSING
    """The slope of the terrain (in radians)."""

    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""

    inverted: bool = False
    """Whether the pyramid is inverted. Defaults to False.

    If True, the terrain is inverted such that the platform is at the bottom and the slopes are upwards.
    """


@configclass
class HfInvertedPyramidSlopedTerrainCfg(HfPyramidSlopedTerrainCfg):
    """Configuration for an inverted pyramid sloped height field terrain.

    Note:
        This is a subclass of :class:`HfPyramidSlopedTerrainCfg` with :obj:`inverted` set to True.
        We make it as a separate class to make it easier to distinguish between the two and match
        the naming convention of the other terrains.
    """

    inverted: bool = True


@configclass
class HfPyramidStairsTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a pyramid stairs height field terrain."""

    function = hf_terrains.pyramid_stairs_terrain

    step_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the steps (in m)."""

    step_width: float = MISSING
    """The width of the steps (in m)."""

    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""

    inverted: bool = False
    """Whether the pyramid stairs is inverted. Defaults to False.

    If True, the terrain is inverted such that the platform is at the bottom and the stairs are upwards.
    """


@configclass
class HfInvertedPyramidStairsTerrainCfg(HfPyramidStairsTerrainCfg):
    """Configuration for an inverted pyramid stairs height field terrain.

    Note:
        This is a subclass of :class:`HfPyramidStairsTerrainCfg` with :obj:`inverted` set to True.
        We make it as a separate class to make it easier to distinguish between the two and match
        the naming convention of the other terrains.
    """

    inverted: bool = True


@configclass
class HfDiscreteObstaclesTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a discrete obstacles height field terrain."""

    function = hf_terrains.discrete_obstacles_terrain

    obstacle_height_mode: str = "choice"
    """The mode to use for the obstacle height. Defaults to "choice".

    The following modes are supported: "choice", "fixed".
    """

    obstacle_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the obstacles (in m)."""

    obstacle_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the obstacles (in m)."""

    num_obstacles: int = MISSING
    """The number of obstacles to generate."""

    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""


@configclass
class HfWaveTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a wave height field terrain."""

    function = hf_terrains.wave_terrain

    amplitude_range: tuple[float, float] = MISSING
    """The minimum and maximum amplitude of the wave (in m)."""

    num_waves: int = 1
    """The number of waves to generate. Defaults to 1."""


@configclass
class HfSteppingStonesTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a stepping stones height field terrain."""

    function = hf_terrains.stepping_stones_terrain

    stone_height_max: float = MISSING
    """The maximum height of the stones (in m)."""

    stone_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the stones (in m)."""

    stone_distance_range: tuple[float, float] = MISSING
    """The minimum and maximum distance between stones (in m)."""

    holes_depth: float = -10.0
    """The depth of the holes (negative obstacles). Defaults to -10.0."""

    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""


@configclass
class HfSteppingSlopeStonesTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a stepping stones height field terrain."""

    function = hf_terrains.stepping_slope_stones_terrain

    stone_height_max: float = MISSING
    """The maximum height of the stones (in m)."""

    stone_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the stones (in m)."""

    stone_distance_range: tuple[float, float] = MISSING
    """The minimum and maximum distance between stones (in m)."""

    holes_depth: float = -10.0
    """The depth of the holes (negative obstacles). Defaults to -10.0."""

    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""

    stone_slope_range: tuple[float, float] = (0.0, 0.2)  # 新增：石头斜度范围（坡度，以高度差表示）


@configclass
class ImageHeightLayerCfg:
    """Configuration for a single image-based height layer."""

    image_path: str = MISSING
    """Path to the image file for this layer."""

    height_scale: float = 0.1
    """Height scaling factor for this layer. Multiplies the extracted height values."""

    blend_mode: Literal["add", "multiply", "max", "min"] = "add"
    """Blending mode for combining this layer with other layers.
    
    Available modes:
    - "add": Add layer heights to accumulated heights (default)
    - "multiply": Multiply accumulated heights by (1 + layer_height)
    - "max": Use maximum of accumulated and layer heights
    - "min": Use minimum of accumulated and layer heights
    """

    color_mode: Literal["grayscale", "hsv"] = "hsv"
    """Color mode for extracting height information from the image.
    
    Available modes:
    - "grayscale": Convert to grayscale and use luminance (default)
    - "hsv": Use hue value from HSV color space (0-360 degrees)
    """

    smoothing_sigma: float = 0.0
    """Smoothing sigma (radius) for this layer's height field in pixels. Defaults to 0.0 (no smoothing).
    
    When non-zero, applies Gaussian smoothing to the layer's height field before blending,
    which can help reduce jagged edges and artifacts from image-based terrain generation.
    """


@configclass
class HfImageBasedTerrainCfg(HfTerrainBaseCfg):
    """Configuration for image-based height field terrain generation.

    This configuration allows generating terrains from image files, where pixel colors
    are mapped to terrain heights. Supports both single-image and multi-layer blending modes.

    Note:
        Image dimensions are automatically validated against the maximum allowed size
        calculated from terrain size and horizontal scale. Images must not exceed these limits.
    """

    function = hf_terrains.image_based_terrain

    # Single image mode (backward compatible)
    image_path: str | None = None
    """Path to a single image file. Use this instead of layer_configs for simple terrains."""

    height_scale: float = 0.1
    """Global height scaling factor. Multiplies the color values to get actual height."""

    color_mode: Literal["grayscale", "hsv"] = "hsv"
    """Color mode for extracting height information from images.
    
    Available modes:
    - "grayscale": Convert to grayscale and use luminance (default)
    - "hsv": Use hue value from HSV color space (0-360 degrees, suitable for heatmaps)
    """

    # Multi-layer blending mode
    layer_configs: list[ImageHeightLayerCfg] | None = None
    """List of image layer configurations for multi-layer blending.
    Use this instead of image_path for complex terrains with multiple layers."""

    grayscale_zero_point: float = 127.5
    """The grayscale value that corresponds to zero height when using grayscale color mode. Defaults to 127.5.

    When using grayscale color mode, this value determines which brightness corresponds to the baseline terrain height.
    Values above this point represent elevated terrain (positive heights), while values below
    represent depressed terrain (negative heights).

    Example with grayscale_zero_point=127.5:
    - Value 0 (black): Maximum depression
    - Value 63.75: Medium depression
    - Value 127.5: Zero height (baseline)
    - Value 191.25: Medium elevation
    - Value 255 (white): Maximum elevation
    """
    
    hsv_zero_point: float = 180.0
    """The hue value that corresponds to zero height when using HSV color mode. Defaults to 180.0.

    When using HSV color mode, this value determines which hue corresponds to the baseline terrain height.
    Hue values above this point represent elevated terrain (positive heights), while values below
    represent depressed terrain (negative heights).

    Example with hsv_zero_point=180.0:
    - Hue 0° (red): Maximum depression
    - Hue 120° (green): Medium depression
    - Hue 180° (cyan): Zero height (baseline)
    - Hue 240° (blue): Medium elevation
    - Hue 360° (red): Maximum elevation
    """
    
    invert_height: bool = False
    """If True, invert the height values (white becomes low, black becomes high)."""
    
    platform_width: float = 1.0
    """Width of the flat platform at the center of the terrain (in m). Defaults to 1.0.
    Set to 0.0 to disable platform."""

    terrain_smoothing_sigma: float = 0.0
    """Smoothing sigma (radius) for the final combined terrain in pixels. Defaults to 0.0 (no smoothing).
    
    When non-zero, applies Gaussian smoothing to the final combined terrain before adding platform.
    This helps reduce jagged edges and artifacts from the image-based terrain generation.
    """

    resize_to_target: bool = False
    """If True, resize the image to match the target terrain dimensions using interpolation.
    If False (default), center the original image in the target terrain dimensions when image is smaller.
    """

    resize_interpolation: Literal["nearest", "bilinear", "bicubic", "lanczos"] = "bilinear"
    """Interpolation method to use when resize_to_target is True. Defaults to "bilinear".

    Available modes:
    - "nearest": Nearest neighbor interpolation
    - "bilinear": Bilinear interpolation (default)
    - "bicubic": Bicubic interpolation
    - "lanczos": Lanczos interpolation (highest quality, slower)
    """

    save_resized_image: bool = False
    """If True, save the resized image to a file when resize_to_target is True.
    The saved image will have the same name as the original but with '_resize' suffix added before the extension.
    Only applicable when resize_to_target is True. Defaults to False.
    """

