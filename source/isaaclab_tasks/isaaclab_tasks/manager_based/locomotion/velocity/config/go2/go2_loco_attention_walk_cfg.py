# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


# ============================================================================================================
# 导入资源
import math
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg, patterns, RayCasterCfg, RayCasterColorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

##
# Pre-defined configs - Rough Terrain
##
from isaaclab.terrains.config.rough import ATTENTION_WALK_TERRAINS_S1_CFG  # isort: skip
from isaaclab.terrains.config.rough import ATTENTION_WALK_TERRAINS_S2_CFG  # isort: skip

##
# Pre-defined configs - Unitree Go2
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_ESC_CFG  # isort: skip

# ============================================================================================================
# 定义交互场景 CFG


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # 地形
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        # terrain_generator=ATTENTION_WALK_TERRAINS_S1_CFG,   # stage1 enbale
        terrain_generator=ATTENTION_WALK_TERRAINS_S2_CFG,   # stage2 enbale
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # 机器人
    robot: ArticulationCfg = UNITREE_GO2_ESC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 传感器(虚拟)
    # 接触力传感器
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # IMU
    base_imu = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/base", offset=ImuCfg.OffsetCfg(pos=(-0.02557, 0.0, 0.04232)), debug_vis=False)
    # 高度扫描仪
    base_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        max_distance=100.0,
        ray_alignment="yaw",
        pattern_cfg=patterns.SingleRayPatternCfg(direction=(0.0, 0.0, -1.0)),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    FL_foot_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/FL_foot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        max_distance=100.0,
        ray_alignment="yaw",
        pattern_cfg=patterns.SingleRayPatternCfg(direction=(0.0, 0.0, -1.0)),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    FR_foot_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/FR_foot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        max_distance=100.0,
        ray_alignment="yaw",
        pattern_cfg=patterns.SingleRayPatternCfg(direction=(0.0, 0.0, -1.0)),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    RL_foot_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RL_foot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        max_distance=100.0,
        ray_alignment="yaw",
        pattern_cfg=patterns.SingleRayPatternCfg(direction=(0.0, 0.0, -1.0)),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    RR_foot_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RR_foot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        max_distance=100.0,
        ray_alignment="yaw",
        pattern_cfg=patterns.SingleRayPatternCfg(direction=(0.0, 0.0, -1.0)),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    # 地图扫描仪
    base_map_scanner = RayCasterColorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterColorCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        max_distance=100.0,
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    # 足端边缘检测器
    FL_foot_edge_detecter = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/FL_foot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        max_distance=100.0,
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.035, size=(0.07, 0.07)),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    FR_foot_edge_detecter = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/FR_foot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        max_distance=100.0,
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.035, size=(0.07, 0.07)),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    RL_foot_edge_detecter = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RL_foot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        max_distance=100.0,
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.035, size=(0.07, 0.07)),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    RR_foot_edge_detecter = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RR_foot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        max_distance=100.0,
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.035, size=(0.07, 0.07)),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )

    # 光源
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# ============================================================================================================
# 定义命令 CFG


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # # stage1 enable ===================================================================================================
    # # stage2-1 enable ===================================================================================================
    # base_velocity = mdp.UniformVelocityCommandCfgUser(
    #     asset_name="robot",
    #     resampling_time_range=(10.0, 20.0),
    #     rel_standing_envs=0.05,
    #     rel_vel_world_envs=1.0,
    #     heading_control_stiffness=0.5,
    #     debug_vis=True,
    #     ranges=mdp.UniformVelocityCommandCfgUser.Ranges(
    #         lin_vel_x=(0.5, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.5, 0.5), heading=(-0.0, 0.0)
    #     ),
    # )

    # stage2-2 enable ===================================================================================================
    base_velocity = mdp.UniformVelocityCommandCfgUser(
        asset_name="robot",
        resampling_time_range=(10.0, 20.0),
        rel_standing_envs=0.05,
        rel_vel_world_envs=1.0,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfgUser.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-0.5, 0.5), heading=(-0.0, 0.0)
        ),
    )


# ============================================================================================================
# 定义动作 CFG


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=[".*"], 
        scale={
            ".*_hip_joint": 0.125,
            ".*_thigh_joint": 0.25,
            ".*_calf_joint": 0.25
        }, 
        use_default_offset=True, 
        clip={".*": (-5.0, 5.0)}
    )


# ============================================================================================================
# 定义观测 CFG


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    # ------------------------------------------------------------------------------------------------------------------
    @configclass
    class Proprioception(ObsGroup):
        """Observations for policy group."""

        velocity_commands = ObsTerm(func=mdp.generated_commands, scale=2.0, clip=(-5.0, 5.0), params={"command_name": "base_velocity"})
        imu_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel, 
            params={"asset_cfg": SceneEntityCfg("base_imu")}, 
            scale=0.25, 
            clip=(-100.0, 100.0),
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            scale=2.0,
            clip=(-1.0, 1.0),
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel, 
            scale=2.0, 
            clip=(-5.0, 5.0),
            noise=NoiseModelWithAdditiveBiasCfg(
                noise_cfg=Unoise(operation="add", n_min=-0.005, n_max=0.005),
                bias_noise_cfg=Unoise(operation="abs", n_min=-0.005, n_max=0.005),
                sample_bias_per_component=True,
            )
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, 
            scale=0.1, 
            clip=(-100.0, 100.0),
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(func=mdp.last_action, scale=2.0, clip=(-5.0, 5.0))

        def __post_init__(self):
            # self.enable_corruption = False  # stage1 enable - 关闭噪声
            self.enable_corruption = True   # stage2 enable - 启用噪声
            self.concatenate_terms = True   
            self.history_length = 3        

    proprioception: Proprioception = Proprioception()

    # ------------------------------------------------------------------------------------------------------------------
    @configclass
    class MapScans(ObsGroup):
        """Observations for scan group."""

        height_scan_3d = ObsTerm(
            func=mdp.height_scan_3d,
            params={"sensor_cfg": SceneEntityCfg("base_map_scanner"), "offset": 0.0},
            scale=2.0,
            clip=(-1.5, 1.5),
            noise=NoiseModelWithAdditiveBiasCfg(
                noise_cfg=Unoise(operation="add", n_min=-0.015, n_max=0.015),
                bias_noise_cfg=Unoise(operation="abs", n_min=-0.01, n_max=0.01),
                sample_bias_per_component=False,
            )
        )

        def __post_init__(self):
            # self.enable_corruption = False  # stage1 enable - 关闭噪声
            self.enable_corruption = True   # stage2 enable - 启用噪声
            self.concatenate_terms = True   
            self.history_length = 1       

    # scan observation groups
    mapScans: MapScans = MapScans()

    @configclass
    class MapScansCritic(ObsGroup):
        # 高度扫描
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("base_map_scanner"), "offset": 0.0},
            scale=2.0,
            clip=(-1.5, 1.5),
        )

        def __post_init__(self):
            self.enable_corruption = False  # No Noise
            self.concatenate_terms = True
            self.history_length = 1

    mapScansCritic: MapScansCritic = MapScansCritic()

    # ------------------------------------------------------------------------------------------------------------------
    @configclass
    class Privileged(ObsGroup):
        # 线速度
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0, clip=(-10.0, 10.0))
        # feet distance
        feet_distance = ObsTerm(func=mdp.feet_distance, scale=20.0, clip=(-0.5, 0.5))
        # feet contact mask
        feet_contact_mask = ObsTerm(
            func=mdp.feet_contact_mask, 
            scale=2.0, 
            clip=(-1.0, 1.0),
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), "threshold": 2.0},
        )
        # feet height
        FL_foot_height = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("FL_foot_height_scanner"), "offset": 0.0},
            scale=10.0,
            clip=(-0.0, 0.5),
        )
        FR_foot_height = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("FR_foot_height_scanner"), "offset": 0.0},
            scale=10.0,
            clip=(-0.0, 0.5),
        )
        RL_foot_height = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("RL_foot_height_scanner"), "offset": 0.0},
            scale=10.0,
            clip=(-0.0, 0.5),
        )
        RR_foot_height = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("RR_foot_height_scanner"), "offset": 0.0},
            scale=10.0,
            clip=(-0.0, 0.5),
        )

        def __post_init__(self):
            self.enable_corruption = False  # 特权观测，不添加噪声
            self.concatenate_terms = True
            self.history_length = 3

    privileged: Privileged = Privileged()


# ============================================================================================================
# 定义事件 CFG


@configclass
class EventCfg:
    """Configuration for events."""

    # # stage1 enable ===================================================================================================
    # # startup
    # physics_material = EventTerm(                       # 随机化物理材质
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "static_friction_range": (0.75, 0.85),
    #         "dynamic_friction_range": (0.6, 0.7),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 64,
    #     },
    # )

    # # reset
    # root_state_random = EventTerm(                      # 随机化根状态
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
    #         "velocity_range": {
    #             "x": (-0.25, 0.25),
    #             "y": (-0.25, 0.25),
    #             "z": (-0.2, 0.2),
    #             "roll": (-0.3, 0.3),
    #             "pitch": (-0.3, 0.3),
    #             "yaw": (-0.3, 0.3),
    #         },
    #     },
    # )
    # reset_joints_by_scale = EventTerm(                  # 随机化关节初始位置
    #     func=mdp.reset_joints_by_scale,
    #     mode="reset",
    #     params={
    #         "position_range": (0.7, 1.3),
    #         "velocity_range": (0.0, 0.0),
    #     },
    # )
    # robot_joint_stiffness_and_damping = EventTerm(      # 随机化关节刚度和阻尼
    #     func=mdp.randomize_actuator_gains,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    #         "stiffness_distribution_params": (0.85, 1.15),
    #         "damping_distribution_params": (0.8, 1.2),
    #         "operation": "scale",
    #         "distribution": "log_uniform",
    #     },
    # )
    # add_base_mass = EventTerm(                          # 随机化根体质量
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #         "mass_distribution_params": (-1.0, 1.0),
    #         "operation": "add",
    #     },
    # )
    # random_base_com = EventTerm(                        # 随机化根体重心
    #     func=mdp.randomize_rigid_body_com,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #         "com_range": {
    #             "x": (-0.07, 0.03),
    #             "y": (-0.02, 0.02),
    #             "z": (0.00, 0.08),
    #         },
    #     },
    # )

    # # interval
    # push_jump = EventTerm(                              # 随机化推动跳跃
    #     func=mdp.push_when_still_stucked_random,
    #     mode="interval",
    #     interval_range_s=(1.0, 1.0),
    #     params={
    #         "command_name": "base_velocity",
    #         "vel_diff_threshold": 0.3,
    #         "stucked_counter_cnt": 3,
    #         "velocity_range": {
    #             "x": (1.0, 2.0), 
    #             "y": (0.0, 0.0), 
    #             "z": (1.0, 2.0),
    #             "roll": (0.0, 0.0), 
    #             "pitch": (0.0, 0.0), 
    #             "yaw":(0.0, 0.0), 
    #         }
    #     },
    # )

    # stage2 enable ===================================================================================================
    # startup
    physics_material = EventTerm(                       # 随机化物理材质
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 0.9),
            "dynamic_friction_range": (0.6, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # reset
    root_state_random = EventTerm(                      # 随机化根状态
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (-0.25, 0.25),
                "y": (-0.25, 0.25),
                "z": (-0.2, 0.2),
                "roll": (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
                "yaw": (-0.3, 0.3),
            },
        },
    )
    reset_joints_by_scale = EventTerm(                  # 随机化关节初始位置
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.7, 1.3),
            "velocity_range": (0.0, 0.0),
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(      # 随机化关节刚度和阻尼
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.25),
            "damping_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    mul_hip_mass = EventTerm(                           # 随机化髋关节质量
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_hip"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    mul_thigh_mass = EventTerm(                         # 随机化大腿质量
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_thigh"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    mul_calf_mass = EventTerm(                          # 随机化小腿质量
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_calf"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    add_base_mass = EventTerm(                          # 随机化根体质量
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
        },
    )
    random_base_com = EventTerm(                        # 随机化根体重心
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {
                "x": (-0.07, 0.03),
                "y": (-0.03, 0.03),
                "z": (0.00, 0.08),
            },
        },
    )

    # # stage1, stage2-1 enable; stage2-2 cancel
    # # interval
    # push_jump = EventTerm(                              # 随机化推动跳跃
    #     func=mdp.push_when_still_stucked_random,
    #     mode="interval",
    #     interval_range_s=(1.0, 1.0),
    #     params={
    #         "command_name": "base_velocity",
    #         "vel_diff_threshold": 0.3,
    #         "stucked_counter_cnt": 3,
    #         "velocity_range": {
    #             "x": (1.0, 2.0), 
    #             "y": (0.0, 0.0), 
    #             "z": (1.0, 2.0),
    #             "roll": (0.0, 0.0), 
    #             "pitch": (0.0, 0.0), 
    #             "yaw":(0.0, 0.0), 
    #         }
    #     },
    # )

# ============================================================================================================
# 定义奖励 CFG


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    # 线速度跟随奖励 [任务]
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=4.0, params={"command_name": "base_velocity", "std": math.sqrt(1.0)}
    )
    # 角速度跟随奖励 [任务]
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=2.0, params={"command_name": "base_velocity", "std": math.sqrt(0.5)}
    )
    # gallop gait - stage2 cancel
    gallop_gait = RewTerm(
        func=mdp.gallop_gait_v2,
        weight=1.2,
        params={
            "command_name": "base_velocity", 
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), 
            "force_threshold": 5.0,
            "sync_std": math.sqrt(0.4),
            "target_contact_duration": 0.3,
            "duration_std": math.sqrt(0.5),
        },
    )

    # -- penalties
    # termination penalty
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)
    # 基座高度偏离惩罚 [姿态]
    base_height_l2 = RewTerm(
        func=mdp.base_height_to_feet_range_l1,
        weight=-3.0,
        # params={"height_gap_target": 0.3, "diff_range": 0.05, "asset_cfg": SceneEntityCfg("robot", body_names="base")},
        params={"height_gap_target": 0.33, "diff_range": 0.05, "asset_cfg": SceneEntityCfg("robot", body_names="base")},
    )
    # 默认站立姿态 [姿态]
    default_stand_pos = RewTerm(
        func=mdp.default_stand_pose,
        weight=-1.0,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    )
    # 前/后向运动时，对hip关节的软限制 [姿态]
    hip_pos_fb_limits = RewTerm(
        func=mdp.hip_joint_pos_fb_limits,
        weight=-1.5,
        params={"command_name": "base_velocity", "hip_limit_pos": -0.25, "asset_cfg": SceneEntityCfg("robot")},
    )
    # 站立速度惩罚 [能量]
    stand_still_vel = RewTerm(func=mdp.stand_still_vel, weight=-0.025, params={"command_name": "base_velocity"})
    # 关节力矩惩罚 [能量]
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-5)
    # 关节加速度惩罚 [平滑]
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.0e-7)
    # 动作频率惩罚 [平滑]
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_limit, weight=-0.005)
    # 关节软限制 [姿态]
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.05)
    # feet slide
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.15,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        }
    )
    # contact force penalty
    contact_force_penalty = RewTerm(
        func=mdp.contact_forces,
        weight=-2.0e-5,
        params={"threshold": 200.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    # 接触惩罚 [姿态]
    undesired_contacts_head = RewTerm(
        func=mdp.undesired_contacts,
        weight=-20.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="Head.*"), "threshold": 1.0},
    )
    undesired_contacts_base = RewTerm(
        func=mdp.undesired_contacts,
        weight=-5.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 30.0},
    )
    undesired_contacts_thigh = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"), "threshold": 1.0},
    )
    undesired_contacts_calf = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"), "threshold": 1.0},
    )
    # x 轴角速度惩罚 [姿态]
    ang_vel_x_l2 = RewTerm(func=mdp.ang_vel_x_l2, weight=-0.1)
    # 踏足边缘惩罚
    foot_edge = RewTerm(
        func=mdp.foot_edge_contact,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), 
            "contact_threshold": 5.0,
            "FL_foot_edge_detecter": SceneEntityCfg("FL_foot_edge_detecter"),
            "FR_foot_edge_detecter": SceneEntityCfg("FR_foot_edge_detecter"),
            "RL_foot_edge_detecter": SceneEntityCfg("RL_foot_edge_detecter"),
            "RR_foot_edge_detecter": SceneEntityCfg("RR_foot_edge_detecter"),
            "height_threshold": 0.5,
            "cnt_threshold": 1,
        },
    )
    # feet 同侧足碰撞惩罚
    feet_self_collision = RewTerm(
        func=mdp.feet_self_collision,
        weight=-1.0,
        params={"collision_distance": 0.05},
    )

    # stage2 enbale
    # 基座速度突变惩罚 [平滑]
    base_acc_l2 = RewTerm(
        func=mdp.base_acc_l2,
        weight=-5e-7,
        params={"sensor_cfg": SceneEntityCfg("base_imu")},
    )
    # 姿态不水平惩罚 [姿态]
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)
    # y 轴角速度惩罚
    ang_vel_y_l2 = RewTerm(func=mdp.ang_vel_y_l2, weight=-0.01)


# ============================================================================================================
# 定义终止条件 CFG


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.pi*80.0/180.0},
    )


# ============================================================================================================
# 定义课程 CFG


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


# ============================================================================================================
# 定义环境管理器 CFG


@configclass
class Go2LocomotionSkillEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # 交互场景类实例化
    scene: MySceneCfg = MySceneCfg(num_envs=16, env_spacing=2.5)

    # 观测类实例化
    observations: ObservationsCfg = ObservationsCfg()
    # 动作类实例化
    actions: ActionsCfg = ActionsCfg()
    # 命令类实例化
    commands: CommandsCfg = CommandsCfg()

    # MDP 相关
    # 奖励类实例化
    rewards: RewardsCfg = RewardsCfg()
    # 终止条件类实例化
    terminations: TerminationsCfg = TerminationsCfg()
    # 事件类实例化
    events: EventCfg = EventCfg()
    # 课程类实例化
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # 场景仿真基本设定
        self.decimation = 4  # 场景仿真频率 = sim_dt/4 = 50 Hz
        self.episode_length_s = 20.0  # 截断条件：时间 20s

        # 物理仿真基本设定
        self.sim.dt = 0.005  # 物理仿真频率 = 200 Hz
        self.sim.render_interval = self.decimation  # 渲染频率 = 场景仿真频率 = 50Hz
        self.sim.disable_contact_processing = True  # 禁用接触联系处理
        self.sim.physics_material = self.scene.terrain.physics_material  # 指定刚体的默认物理材质设置
        self.sim.physx.gpu_max_rigid_patch_count = 20 * 2**17

        # 修改传感器更新频率
        # 我们根据最小更新周期（物理更新周期）勾选所有传感器
        if self.scene.contact_forces is not None:  # 接触力传感器
            self.scene.contact_forces.update_period = self.sim.dt  # 200 Hz
        if self.scene.base_imu is not None:  # IMU
            self.scene.base_imu.update_period = self.sim.dt  # 200 Hz
        if self.scene.base_height_scanner is not None:  # base 单点高度扫描
            self.scene.base_height_scanner.update_period = self.decimation * self.sim.dt  # 50 Hz
        if self.scene.FL_foot_height_scanner is not None:  # FL 足端单点高度扫描
            self.scene.FL_foot_height_scanner.update_period = self.decimation * self.sim.dt  # 50 Hz
        if self.scene.FR_foot_height_scanner is not None:  # FR 足端单点高度扫描
            self.scene.FR_foot_height_scanner.update_period = self.decimation * self.sim.dt  # 50 Hz
        if self.scene.RL_foot_height_scanner is not None:  # RL 足端单点高度扫描
            self.scene.RL_foot_height_scanner.update_period = self.decimation * self.sim.dt  # 50 Hz
        if self.scene.RR_foot_height_scanner is not None:  # RR 足端单点高度扫描
            self.scene.RR_foot_height_scanner.update_period = self.decimation * self.sim.dt  # 50 Hz
        if self.scene.base_map_scanner is not None:  # base 高度扫描
            self.scene.base_map_scanner.update_period = self.decimation * self.sim.dt  # 50 Hz
        if self.scene.FL_foot_edge_detecter is not None:  # FL 足端边缘检测
            self.scene.FL_foot_edge_detecter.update_period = self.decimation * self.sim.dt  # 50 Hz
        if self.scene.FR_foot_edge_detecter is not None:  # FR 足端边缘检测
            self.scene.FR_foot_edge_detecter.update_period = self.decimation * self.sim.dt  # 50 Hz
        if self.scene.RL_foot_edge_detecter is not None:  # RL 足端边缘检测
            self.scene.RL_foot_edge_detecter.update_period = self.decimation * self.sim.dt  # 50 Hz
        if self.scene.RR_foot_edge_detecter is not None:  # RR 足端边缘检测
            self.scene.RR_foot_edge_detecter.update_period = self.decimation * self.sim.dt  # 50 Hz

        # 检查地形等级&课程学习是否设定启用
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            # 若启用，且地形生成器已指定
            if self.scene.terrain.terrain_generator is not None:
                # 则使能地形生成器的课程学习(生成地形从易到难)
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


# ============================================================================================================
# 定义环境管理器 CFG - Play 播放专用


class Go2LocomotionSkillEnvCfg_Play(Go2LocomotionSkillEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # 小规模播放
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.episode_length_s = 30.0

        # 可视化 marker
        self.scene.base_height_scanner.debug_vis = False
        self.scene.FL_foot_height_scanner.debug_vis = False
        self.scene.FR_foot_height_scanner.debug_vis = False
        self.scene.RL_foot_height_scanner.debug_vis = False
        self.scene.RR_foot_height_scanner.debug_vis = False
        self.scene.FL_foot_edge_detecter.debug_vis = False
        self.scene.FR_foot_edge_detecter.debug_vis = False
        self.scene.RL_foot_edge_detecter.debug_vis = False
        self.scene.RR_foot_edge_detecter.debug_vis = False
        self.scene.base_map_scanner.debug_vis = True

        # 限定速度指令
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)

        # # 地形课程取消，地形块类型随机排布
        # self.scene.terrain.max_init_terrain_level = None
        # self.scene.terrain.terrain_generator.curriculum = False

        # 移除部分事件
        self.events.push_jump = None

