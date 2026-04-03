# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen

from ..terrain_generator_cfg import TerrainGeneratorCfg

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)
"""Rough terrains configuration."""


# ===================================================================================================================================
# 测试注意力感知模块的训练地形


ATTENTION_WALK_TERRAINS_S1_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=100.0,
    num_rows=10,
    num_cols=11,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.25),
            step_width=0.35,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.25),
            step_width=0.35,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.2, amplitude_range=(0.1, 0.4), num_waves=3, border_width=1.0
        ),
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.2, 0.5), platform_width=2.0, border_width=1.0
        ),
        "pallets": terrain_gen.MeshPalletsTerrainCfg(
            proportion=0.2,
            pit_depth=2.0,
            ring_spacing_range=(0.1, 0.4),
            ring_width_range=(0.3, 0.6),
            ring_height_range=(-0.12, 0.12),
            ring_thickness=0.4,
            platform_width=2.0,
            randomize_heights=True,
            randomize_widths=True,
        ),
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.2,
            stone_height_max=0.05,
            stone_width_range=(0.5, 1.0),
            stone_distance_range=(0.1, 0.5),
            holes_depth=-2.0,
            platform_width=2.0,
            border_width=0.75,
        ),
        "pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.2, 0.5), platform_width=2.0, border_width=1.0
        ),
        "grid": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.025, 0.125), platform_width=2.0
        ),
        "double_pit": terrain_gen.MeshPitTerrainCfg(
            proportion=0.2,
            pit_depth_range=(0.02, 0.25),
            platform_width=2.0,
            double_pit=True,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2,
            noise_range=(-0.05, 0.05),
            noise_step=0.005,
            border_width=1.0,
        ),
        "gap": terrain_gen.MeshGapTerrainCfg(
            proportion=0.2,
            gap_width_range=(0.1, 0.5),
            platform_width=2.0,
        ),
    },
)
"""Rough terrains configuration."""


ATTENTION_WALK_TERRAINS_S2_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=100.0,
    num_rows=10,
    num_cols=17,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.35),
            step_width=0.35,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.35),
            step_width=0.35,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.2, amplitude_range=(0.1, 0.4), num_waves=3, border_width=1.0
        ),
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.2, 0.5), platform_width=2.0, border_width=1.0
        ),
        "pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.2, 0.5), platform_width=2.0, border_width=1.0
        ),
        "pallets": terrain_gen.MeshPalletsTerrainCfg(
            proportion=0.2,
            pit_depth=2.0,
            ring_spacing_range=(0.1, 0.65),
            ring_width_range=(0.3, 0.6),
            ring_height_range=(-0.15, 0.15),
            ring_thickness=0.4,
            platform_width=2.0,
            randomize_heights=True,
            randomize_widths=True,
        ),
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.2,
            stone_height_max=0.1,
            stone_width_range=(0.5, 1.0),
            stone_distance_range=(0.1, 0.5),
            holes_depth=-2.0,
            platform_width=2.0,
            border_width=0.75,
        ),
        "steeping_slope_stones": terrain_gen.HfSteppingSlopeStonesTerrainCfg(
            proportion=0.2,
            stone_height_max=0.05,
            stone_width_range=(0.65, 1.0),
            stone_distance_range=(0.1, 0.45),
            holes_depth=-2.0,
            platform_width=2.0,
            border_width=0.75,
            stone_slope_range=(0.05, 0.3),
        ),
        "mesh_stepping_stones": terrain_gen.MeshSteppingStonesTerrainCfg(
            proportion=0.2,
            stone_params_start=terrain_gen.MeshSteppingStonesTerrainCfg.StoneCfg(
                width=1.0,
                spacing=1.25,
                height=2.0,
                max_tilt_angle=0.06,
            ),
            stone_params_end=terrain_gen.MeshSteppingStonesTerrainCfg.StoneCfg(
                width=0.5,
                spacing=0.6,
                height=2.0,
                max_tilt_angle=0.03,
            ),
            obstacle_num_range=(4, 8),
            obstacle_size_range=(0.6, 0.4),
            obstacle_height_scale=3.0,
            obstacle_max_tilt_angle=0.1,
            pit_depth=2.0,
            height_variation=0.4,
            platform_width=2.0,
            platform_height=2.3,
            stone_types=["box", "cylinder"],
        ),
        "bars": terrain_gen.MeshPlatformBarsTerrainCfg(
            proportion=0.2,
            platform_width=2.0,
            pit_depth=2.0,
            bars_num_range=(16, 32),
            bar_width_range=(0.3, 0.6),
            border_width=0.8,
        ),
        "grid": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.025, 0.125), platform_width=2.0
        ),
        "double_pit": terrain_gen.MeshPitTerrainCfg(
            proportion=0.2,
            pit_depth_range=(0.02, 0.4),
            platform_width=2.0,
            double_pit=True,
        ),
        "double_box": terrain_gen.MeshBoxTerrainCfg(
            proportion=0.2,
            box_height_range=(0.02, 0.8),
            platform_width=2.0,
            double_box=True,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2,
            noise_range=(-0.05, 0.05),
            noise_step=0.005,
            border_width=1.0,
        ),
        "gap": terrain_gen.MeshGapTerrainCfg(
            proportion=0.2,
            gap_width_range=(0.1, 0.65),
            platform_width=2.0,
        ),
        "star": terrain_gen.MeshStarTerrainCfg(
            proportion=0.2,
            num_bars=6,
            bar_width_range=(0.6, 1.5),
            bar_height_range=(0.5, 2.0),
            platform_width=2.0,
        ),
        "star_inv": terrain_gen.MeshStarInvTerrainCfg(
            proportion=0.2,
            num_bars=6,
            bar_width_range=(0.6, 1.5),
            bar_height_range=(0.5, 2.0),
            platform_width=2.0,
        ),
    },
)
"""Rough terrains configuration."""


# ===================================================================================================================================
BLIND_WALK_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=100.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.2, amplitude_range=(0.1, 0.3), num_waves=2.0, border_width=1.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2,
            noise_range=(-0.03, 0.03),
            noise_step=0.01,
            border_width=1.0,
        ),
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.1, 0.4), platform_width=2.0, border_width=1.0
        ),
        "pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.1, 0.4), platform_width=2.0, border_width=1.0
        ),
    },
)



SKILL_WALK_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=100.0,
    num_rows=10,
    num_cols=16,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=1.2,
    use_cache=False,
    sub_terrains={
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.2, amplitude_range=(0.1, 0.4), num_waves=3.0, border_width=1.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2,
            noise_range=(-0.06, 0.06),
            noise_step=0.005,
            border_width=1.0,
        ),
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.3, 0.5), platform_width=2.0, border_width=1.0
        ),
        "pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.3, 0.5), platform_width=2.0, border_width=1.0
        ),
        "grid": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.025, 0.1), platform_width=2.0
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.2),
            step_width=0.35,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.2),
            step_width=0.35,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.2,
            obstacle_width_range=(0.2, 1.2),
            obstacle_height_range=(0.05, 0.15),
            num_obstacles=60,
            platform_width=2.0,
            border_width=1.0,
        ),
    },
)


SKILL_WALK_PLUS_TERRAINS_S1_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=100.0,
    num_rows=10,
    num_cols=12,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=1.2,
    use_cache=False,
    sub_terrains={
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.2, amplitude_range=(0.1, 0.4), num_waves=3.0, border_width=1.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2,
            noise_range=(-0.06, 0.06),
            noise_step=0.005,
            border_width=1.0,
        ),
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.2, 0.5), platform_width=2.0, border_width=1.0
        ),
        "pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.2, 0.5), platform_width=2.0, border_width=1.0
        ),
        "grid": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.025, 0.125), platform_width=2.0
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.2),
            step_width=0.35,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.2),
            step_width=0.35,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.2,
            obstacle_width_range=(0.2, 1.2),
            obstacle_height_range=(0.05, 0.15),
            num_obstacles=60,
            platform_width=2.0,
            border_width=1.0,
        ),
    },
)


SKILL_WALK_PLUS_TERRAINS_S2_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=100.0,
    num_rows=10,
    num_cols=12,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=1.2,
    use_cache=False,
    sub_terrains={
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.2, amplitude_range=(0.1, 0.4), num_waves=3.0, border_width=1.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2,
            noise_range=(-0.06, 0.06),
            noise_step=0.005,
            border_width=1.0,
        ),
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.2, 0.5), platform_width=2.0, border_width=1.0
        ),
        "pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.2, 0.5), platform_width=2.0, border_width=1.0
        ),
        "grid": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.025, 0.125), platform_width=2.0
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.2),
            step_width=0.35,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.2),
            step_width=0.35,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.2,
            obstacle_width_range=(0.2, 1.2),
            obstacle_height_range=(0.05, 0.15),
            num_obstacles=60,
            platform_width=2.0,
            border_width=1.0,
        ),
        # 额外的障碍地形，增强地形多样性，应对更复杂的地形
        "star": terrain_gen.MeshStarTerrainCfg(
            proportion=0.2,
            num_bars=6,
            bar_width_range=(0.8, 1.5),
            bar_height_range=(0.5, 2.0),
            platform_width=2.0,
        ),
        "star_inv": terrain_gen.MeshStarInvTerrainCfg(
            proportion=0.2,
            num_bars=6,
            bar_width_range=(0.8, 1.5),
            bar_height_range=(0.5, 2.0),
            platform_width=2.0,
        ),
        "discrete_obstacles_hard_small": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.2,
            obstacle_width_range=(0.4, 1.5),
            obstacle_height_range=(0.4, 2.0),
            num_obstacles=10,
            platform_width=2.0,
            border_width=2.0,
        ),
        "discrete_obstacles_hard_big": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.2,
            obstacle_width_range=(0.8, 3.0),
            obstacle_height_range=(1.0, 2.0),
            num_obstacles=5,
            platform_width=2.0,
            border_width=2.0,
        ),
        "pyramid_stairs_holes": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.2),
            step_width=0.35,
            platform_width=2.0,
            border_width=1.0,
            holes=True,
        ),
        "pyramid_stairs_inv_holes": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.2),
            step_width=0.35,
            platform_width=2.0,
            border_width=1.0,
            holes=True,
        ),
        "grid_holes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.025, 0.125), platform_width=2.0, holes=True,
        ),
    },
)


TEST_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=100.0,
    num_rows=5,
    num_cols=5,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=1.2,
    use_cache=False,
    sub_terrains={
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.2,
            stone_height_max=0.12,
            stone_width_range=(0.4, 0.8),
            stone_distance_range=(0.05, 0.2),
            holes_depth=-5.0,
            platform_width=2.0,
        ),
    },
)


