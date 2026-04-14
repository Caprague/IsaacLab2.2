# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ray-cast sensor."""

from __future__ import annotations

import torch
from collections.abc import Callable, Sequence
from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from . import patterns


@configclass
class PatternBaseCfg:
    """Base configuration for a pattern."""

    func: Callable[[PatternBaseCfg, str], tuple[torch.Tensor, torch.Tensor]] = MISSING
    """Function to generate the pattern.

    The function should take in the configuration and the device name as arguments. It should return
    the pattern's starting positions and directions as a tuple of torch.Tensor.
    """


@configclass
class SingleRayPatternCfg(PatternBaseCfg):
    """Configuration for the single ray pattern for ray-casting.

    Defines a single ray in the coordinates of the sensor.

    """

    func: Callable = patterns.single_ray_pattern

    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
    """Ray direction. Defaults to (0.0, 0.0, -1.0)."""


@configclass
class GridPatternCfg(PatternBaseCfg):
    """Configuration for the grid pattern for ray-casting.

    Defines a 2D grid of rays in the coordinates of the sensor.

    .. attention::
        The points are ordered based on the :attr:`ordering` attribute.

    """

    func: Callable = patterns.grid_pattern

    resolution: float = MISSING
    """Grid resolution (in meters)."""

    size: tuple[float, float] = MISSING
    """Grid size (length, width) (in meters)."""

    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
    """Ray direction. Defaults to (0.0, 0.0, -1.0)."""

    ordering: Literal["xy", "yx"] = "xy"
    """Specifies the ordering of points in the generated grid. Defaults to ``"xy"``.

    Consider a grid pattern with points at :math:`(x, y)` where :math:`x` and :math:`y` are the grid indices.
    The ordering of the points can be specified as "xy" or "yx". This determines the inner and outer loop order
    when iterating over the grid points.

    * If "xy" is selected, the points are ordered with inner loop over "x" and outer loop over "y".
    * If "yx" is selected, the points are ordered with inner loop over "y" and outer loop over "x".

    For example, the grid pattern points with :math:`X = (0, 1, 2)` and :math:`Y = (3, 4)`:

    * "xy" ordering: :math:`[(0, 3), (1, 3), (2, 3), (1, 4), (2, 4), (2, 4)]`
    * "yx" ordering: :math:`[(0, 3), (0, 4), (1, 3), (1, 4), (2, 3), (2, 4)]`
    """


@configclass
class Mid360PatternCfg(PatternBaseCfg):
    """Configuration for the Livox Mid-360 LiDAR pattern for ray-casting.

    The Livox Mid-360 is a 360-degree mechanical LiDAR that outputs 200,000 points per second
    at a typical frequency of 10Hz, resulting in 20,000 points per scan.

    The pattern reads pre-recorded scan data from a CSV file containing Azimuth and Zenith angles.

    For dynamic scanning (simulating continuous LiDAR operation), use this pattern with
    :class:`~isaaclab.sensors.ray_caster.ray_caster_lidar.RayCaster` and set
    ``dynamic_pattern=True`` in the sensor configuration. This will cause the pattern to cycle
    through different scan patterns on each sensor update, simulating the mechanical rotation
    of the LiDAR.

    Example:

    .. code-block:: python

        from isaaclab.sensors import RayCasterCfg
        from isaaclab.sensors.patterns import Mid360PatternCfg

        mid360_scanner = RayCasterCfg(
            pattern_cfg=Mid360PatternCfg(
                csv_file_path="/path/to/mid360.csv"
            ),
            dynamic_pattern=True,  # Enable dynamic scanning
            update_period=0.1,  # 10 Hz update rate
            # ... other configurations
        )
    """

    func: Callable = patterns.mid360_pattern

    csv_file_path: str = MISSING
    """Path to the CSV file containing Mid-360 scan data.

    The CSV file should have three columns:
    - Time/s: Timestamp (unused for pattern generation)
    - Azimuth/deg: Horizontal angle in degrees (0-360)
    - Zenith/deg: Vertical angle in degrees (37.836-97.2123)
    """

    points_per_scan: int = 20000
    """Number of points per scan. Defaults to 20000.

    Calculated as: 200,000 points/second / 10Hz = 20,000 points/scan
    """

    total_points: int = 800000
    """Total number of points in the CSV file. Defaults to 800000.

    Used for cyclic reading of scan data across multiple scans.
    """
    
    _scan_index: int = 0


@configclass
class PinholeCameraPatternCfg(PatternBaseCfg):
    """Configuration for a pinhole camera depth image pattern for ray-casting.

    .. caution::
        Focal length as well as the aperture sizes and offsets are set as a tenth of the world unit. In our case, the
        world unit is meters, so all of these values are in cm. For more information, please check:
        https://docs.omniverse.nvidia.com/materials-and-rendering/latest/cameras.html
    """

    func: Callable = patterns.pinhole_camera_pattern

    focal_length: float = 24.0
    """Perspective focal length (in cm). Defaults to 24.0cm.

    Longer lens lengths narrower FOV, shorter lens lengths wider FOV.
    """

    horizontal_aperture: float = 20.955
    """Horizontal aperture (in cm). Defaults to 20.955 cm.

    Emulates sensor/film width on a camera.

    Note:
        The default value is the horizontal aperture of a 35 mm spherical projector.
    """
    vertical_aperture: float | None = None
    r"""Vertical aperture (in cm). Defaults to None.

    Emulates sensor/film height on a camera. If None, then the vertical aperture is calculated based on the
    horizontal aperture and the aspect ratio of the image to maintain squared pixels. In this case, the vertical
    aperture is calculated as:

    .. math::
        \text{vertical aperture} = \text{horizontal aperture} \times \frac{\text{height}}{\text{width}}
    """

    horizontal_aperture_offset: float = 0.0
    """Offsets Resolution/Film gate horizontally. Defaults to 0.0."""

    vertical_aperture_offset: float = 0.0
    """Offsets Resolution/Film gate vertically. Defaults to 0.0."""

    width: int = MISSING
    """Width of the image (in pixels)."""

    height: int = MISSING
    """Height of the image (in pixels)."""

    @classmethod
    def from_intrinsic_matrix(
        cls,
        intrinsic_matrix: list[float],
        width: int,
        height: int,
        focal_length: float = 24.0,
    ) -> PinholeCameraPatternCfg:
        r"""Create a :class:`PinholeCameraPatternCfg` class instance from an intrinsic matrix.

        The intrinsic matrix is a 3x3 matrix that defines the mapping between the 3D world coordinates and
        the 2D image. The matrix is defined as:

        .. math::
            I_{cam} = \begin{bmatrix}
            f_x & 0 & c_x \\
            0 & f_y & c_y \\
            0 & 0 & 1
            \end{bmatrix},

        where :math:`f_x` and :math:`f_y` are the focal length along x and y direction, while :math:`c_x` and :math:`c_y` are the
        principle point offsets along x and y direction respectively.

        Args:
            intrinsic_matrix: Intrinsic matrix of the camera in row-major format.
                The matrix is defined as [f_x, 0, c_x, 0, f_y, c_y, 0, 0, 1]. Shape is (9,).
            width: Width of the image (in pixels).
            height: Height of the image (in pixels).
            focal_length: Focal length of the camera (in cm). Defaults to 24.0 cm.

        Returns:
            An instance of the :class:`PinholeCameraPatternCfg` class.
        """
        # extract parameters from matrix
        f_x = intrinsic_matrix[0]
        c_x = intrinsic_matrix[2]
        f_y = intrinsic_matrix[4]
        c_y = intrinsic_matrix[5]
        # resolve parameters for usd camera
        horizontal_aperture = width * focal_length / f_x
        vertical_aperture = height * focal_length / f_y
        horizontal_aperture_offset = (c_x - width / 2) / f_x
        vertical_aperture_offset = (c_y - height / 2) / f_y

        return cls(
            focal_length=focal_length,
            horizontal_aperture=horizontal_aperture,
            vertical_aperture=vertical_aperture,
            horizontal_aperture_offset=horizontal_aperture_offset,
            vertical_aperture_offset=vertical_aperture_offset,
            width=width,
            height=height,
        )


@configclass
class BpearlPatternCfg(PatternBaseCfg):
    """Configuration for the Bpearl pattern for ray-casting."""

    func: Callable = patterns.bpearl_pattern

    horizontal_fov: float = 360.0
    """Horizontal field of view (in degrees). Defaults to 360.0."""

    horizontal_res: float = 10.0
    """Horizontal resolution (in degrees). Defaults to 10.0."""

    # fmt: off
    vertical_ray_angles: Sequence[float] = [
        89.5, 86.6875, 83.875, 81.0625, 78.25, 75.4375, 72.625, 69.8125, 67.0, 64.1875, 61.375,
        58.5625, 55.75, 52.9375, 50.125, 47.3125, 44.5, 41.6875, 38.875, 36.0625, 33.25, 30.4375,
        27.625, 24.8125, 22, 19.1875, 16.375, 13.5625, 10.75, 7.9375, 5.125, 2.3125
    ]
    # fmt: on
    """Vertical ray angles (in degrees). Defaults to a list of 32 angles.

    Note:
        We manually set the vertical ray angles to match the Bpearl sensor. The ray-angles
        are not evenly spaced.
    """


@configclass
class LidarPatternCfg(PatternBaseCfg):
    """Configuration for the LiDAR pattern for ray-casting."""

    func: Callable = patterns.lidar_pattern

    channels: int = MISSING
    """Number of Channels (Beams). Determines the vertical resolution of the LiDAR sensor."""

    vertical_fov_range: tuple[float, float] = MISSING
    """Vertical field of view range in degrees."""

    horizontal_fov_range: tuple[float, float] = MISSING
    """Horizontal field of view range in degrees."""

    horizontal_res: float = MISSING
    """Horizontal resolution (in degrees)."""


@configclass
class BoxGridPatternCfg(PatternBaseCfg):
    """Configuration for the box grid pattern for ray-casting.

    This pattern creates a 3D box sampling pattern with rays originating from three orthogonal faces
    (e.g., left, top, front) and projecting towards their opposite faces. This enables comprehensive
    3D terrain sampling within a bounding box, capturing vertical structures and occluded areas
    that standard height-field scanning cannot reach.

    The pattern is defined by a bounding box size and resolution. The box is centered at the sensor's
    local origin. The sampling faces are specified by their direction vectors.

    Example:
        Create a 2x2x2 meter box with 0.1 meter resolution:

        .. code-block:: python

            from isaaclab.sensors.patterns import BoxGridPatternCfg

            pattern_cfg = BoxGridPatternCfg(
                resolution=0.1,
                size=(2.0, 2.0, 2.0),  # (length, width, height)
                directions=[(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # x, y, z directions
            )
    """

    func: Callable = patterns.box_grid_pattern

    resolution: float = MISSING
    """Grid resolution (in meters) for the sampling points on each face."""

    size: tuple[float, float, float] = MISSING
    """Box size (length, width, height) in meters. The box is centered at the sensor's origin."""

    directions: Sequence[tuple[float, float, float]] = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    """Sampling directions. Each tuple defines a direction vector for a face.

    Defaults to three orthogonal directions (x, y, z) corresponding to left, front, and top faces.
    For each direction, rays originate from the face opposite to the direction vector and
    project towards the direction.

    For example:
        - (1, 0, 0): Rays start from left face (x = -length/2) and project rightward
        - (0, 1, 0): Rays start from back face (y = -width/2) and project forward
        - (0, 0, 1): Rays start from bottom face (z = -height/2) and project upward
    """

    ordering: Literal["xy", "yx"] = "xy"
    """Specifies the ordering of points in the generated 2D grid on each face.

    See :class:`GridPatternCfg` for details on ordering semantics.
    """
