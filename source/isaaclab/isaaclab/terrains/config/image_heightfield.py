# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen

from ..terrain_generator_cfg import TerrainGeneratorCfg


# ====================================================================================================
# Image-based terrain configurations
# ====================================================================================================


# Single image terrain configuration example
IMAGE_STAIRS_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(15.0, 15.0),
    border_width=100.0,
    num_rows=2,
    num_cols=2,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    # slope_threshold=1.0,
    use_cache=False,
    sub_terrains={
        "stairs_blocks": terrain_gen.HfImageBasedTerrainCfg(
            proportion=1.0,
            image_path="/home/gms/Isaac/IsaacLab2.2/IsaacLab/User/TerrainPics/ConStru_terrain_1.png",
            height_scale=0.01,
            color_mode="hsv",
            hsv_zero_point=180.0,
            invert_height=False,
            platform_width=1.0,
            terrain_smoothing_sigma=0.0,
            resize_to_target=True,
            resize_interpolation="nearest",
            save_resized_image=True,
        ),
    },
)
"""Image-based stairs terrain configuration using single image."""


# Multi-layer terrain configuration example
MULTI_LAYER_TERRAIN_CFG = TerrainGeneratorCfg(
    # size=(15.0, 15.0),
    size=(25.0, 25.0),
    border_width=100.0,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "stairs_blocks": terrain_gen.HfImageBasedTerrainCfg(
            proportion=1.0,
            hsv_zero_point=180.0,
            layer_configs=[
                terrain_gen.ImageHeightLayerCfg(
                    image_path="/home/gms/Isaac/IsaacLab2.2/IsaacLab/User/TerrainPics/layer1_N5.png",
                    height_scale=0.01,
                    blend_mode="add",
                    color_mode="hsv",
                    smoothing_sigma=0.0,
                ),
                terrain_gen.ImageHeightLayerCfg(
                    image_path="/home/gms/Isaac/IsaacLab2.2/IsaacLab/User/TerrainPics/layer2_N5.png",
                    height_scale=0.0075,
                    blend_mode="add",
                    color_mode="hsv",
                    smoothing_sigma=0.0,
                ),
            ],
            platform_width=1.0,
            terrain_smoothing_sigma=0.0,
            resize_to_target=True,
            resize_interpolation="nearest",
            save_resized_image=False,
        ),
    },
)
"""Multi-layer image-based terrain configuration example."""
