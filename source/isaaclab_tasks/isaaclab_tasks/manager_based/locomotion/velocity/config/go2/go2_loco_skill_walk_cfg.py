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
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, ImuCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg, NoiseModelWithPeriodicBiasCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

##
# Pre-defined configs - Rough Terrain
##
from isaaclab.terrains.config.rough import SKILL_WALK_PLUS_TERRAINS_S1_CFG  # isort: skip
from isaaclab.terrains.config.rough import SKILL_WALK_PLUS_TERRAINS_S2_CFG  # isort: skip

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
        terrain_generator=SKILL_WALK_PLUS_TERRAINS_S2_CFG,
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

    # 传感器
    # 接触力传感器
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # IMU
    base_imu = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/base", offset=ImuCfg.OffsetCfg(pos=(-0.02557, 0.0, 0.04232)), debug_vis=False)
    # 高度扫描仪
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        max_distance=100.0,
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    height_scanner_lf = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        max_distance=100.0,
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
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
    # base_velocity = mdp.UniformVelocityCommandCfgUser(
    #     asset_name="robot",
    #     resampling_time_range=(10.0, 20.0),
    #     rel_standing_envs=0.05,
    #     rel_vel_world_envs=1.0,                 # must set to 1.0
    #     heading_control_stiffness=0.5,
    #     debug_vis=True,
    #     ranges=mdp.UniformVelocityCommandCfgUser.Ranges(
    #         lin_vel_x=(0.5, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.5, 0.5), heading=(-0.0, 0.0)
    #     ),
    # )

    # stage2 enable ===================================================================================================
    base_velocity = mdp.UniformVelocityCommandCfgUser(
        asset_name="robot",
        resampling_time_range=(10.0, 20.0),
        rel_standing_envs=0.05,
        rel_vel_world_envs=0.75,              # 0.75 : 0.25
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfgUser.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
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

    # Teacher Observation Groups -----------------------------------------------------------------------------
    @configclass
    class Proprioception(ObsGroup):
        # 周期相位 2
        phase = ObsTerm(func=mdp.cycle_phase, scale=3.0, clip=(-1.0, 1.0), params={"cycle_period": 0.7})
        # 速度指令
        velocity_commands = ObsTerm(func=mdp.generated_commands, scale=2.0, clip=(-5.0, 5.0), params={"command_name": "base_velocity"})
        # 角速度
        imu_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel, 
            params={"asset_cfg": SceneEntityCfg("base_imu")}, 
            scale=0.25, 
            clip=(-100.0, 100.0),
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        # 投影重力加速度
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            scale=2.0,
            clip=(-1.0, 1.0),
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        # 关节位置
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel, 
            scale=2.0, 
            clip=(-5.0, 5.0), 
            noise=NoiseModelWithAdditiveBiasCfg(
                noise_cfg=Unoise(operation="add", n_min=-0.005, n_max=0.005),
                bias_noise_cfg=Unoise(operation="abs", n_min=-0.01, n_max=0.01),
                sample_bias_per_component=True,
            )
        )
        # 关节速度
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, 
            scale=0.1, 
            clip=(-100.0, 100.0), 
            noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        # 动作
        actions = ObsTerm(func=mdp.last_action, scale=2.0, clip=(-5.0, 5.0))

        def __post_init__(self):
            self.enable_corruption = False  # No Noise
            self.concatenate_terms = True
            self.history_length = 5

    proprioception: Proprioception = Proprioception()


    @configclass
    class MapScans(ObsGroup):
        # 高度扫描
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.0},
            clip=(-1.5, 1.5),
            scale=2.0,
        )

        def __post_init__(self):
            self.enable_corruption = False  # No Noise
            self.concatenate_terms = True
            self.history_length = 1

    mapScans: MapScans = MapScans()


    @configclass
    class Privileged(ObsGroup):
        # Trot 步态相位 Mask
        gait_trot_mask = ObsTerm(func=mdp.gait_trot_mask, scale=3.0, clip=(-1.0, 1.0), params={"cycle_period": 0.7})
        # Feet 地面接触 Mask
        feet_contact_mask = ObsTerm(func=mdp.feet_contact_mask, scale=3.0, clip=(-1.0, 1.0), params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), "threshold": 5.0})
        # 线速度
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0, clip=(-10.0, 10.0))
        # feet distance
        feet_distance = ObsTerm(func=mdp.feet_distance, scale=20.0, clip=(-0.5, 0.5))
        # Base height
        base_height = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("base_height_scanner"), "offset": 0.0},
            scale=10.0,
            clip=(-2.0, 2.0),
        )
        # feet height
        FL_foot_height = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("FL_foot_height_scanner"), "offset": 0.0},
            scale=10.0,
            clip=(-0.0, 1.5),
        )
        FR_foot_height = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("FR_foot_height_scanner"), "offset": 0.0},
            scale=10.0,
            clip=(-0.0, 1.5),
        )
        RL_foot_height = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("RL_foot_height_scanner"), "offset": 0.0},
            scale=10.0,
            clip=(-0.0, 1.5),
        )
        RR_foot_height = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("RR_foot_height_scanner"), "offset": 0.0},
            scale=10.0,
            clip=(-0.0, 1.5),
        )

        def __post_init__(self):
            self.enable_corruption = False  # No Noise
            self.concatenate_terms = True
            self.history_length = 3

    privileged: Privileged = Privileged()


    # Student Observation Groups -----------------------------------------------------------------------------
    @configclass
    class ProprioceptionNoised(Proprioception):
        def __post_init__(self):
            self.enable_corruption = True   # Noised
            self.concatenate_terms = True
            self.history_length = 6

    proprioception_noised: ProprioceptionNoised = ProprioceptionNoised()

    @configclass
    class MapScansNoised(ObsGroup):
        # 高度扫描
        height_scan_low_freq = ObsTerm(
            func=mdp.height_scan_delay,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner_lf"), 
                "offset": 0.0,
                "delay_range": (0, 10),  # 0.02 * 10 = 0.2 s
                "resample_interval_range": (0.5, 10.0),
                "dt": 0.02,     # 50 hz
                "downsample_freq_scale": 2, # 50 / 2 = 25 hz
            },
            clip=(-1.5, 1.5),
            scale=2.0,
            noise=NoiseModelWithPeriodicBiasCfg(
                noise_cfg=Unoise(operation="add", n_min=-0.05, n_max=0.05),
                bias_noise_cfg=Unoise(operation="abs", n_min=-0.1, n_max=0.1),
                sample_bias_per_component=False,
                bias_resample_interval=(0.5, 10.0),
                dt=0.02,        # 50 hz
            )
        )

        def __post_init__(self):
            self.enable_corruption = True   # Noised
            self.concatenate_terms = True
            self.history_length = 1

    mapScans_noised: MapScansNoised = MapScansNoised()


# ============================================================================================================
# 定义事件 CFG


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 0.9),                # 重要修改 - 静态摩擦系数随机化，增强鲁棒性
            "dynamic_friction_range": (0.4, 0.9),               # 重要修改 - 动态摩擦系数随机化，增强鲁棒性
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )
    random_base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {
                "x": (-0.05, 0.05),
                "y": (-0.03, 0.03),
                "z": (0.00, 0.12),
            },
        },
    )
    mul_hip_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_hip"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    mul_thigh_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_thigh"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    mul_calf_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_calf"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(                      # 重要修改 - 关节刚度和阻尼随机化, 模拟关节磨损
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

    # reset
    reset_root_state_uniform = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.35, 0.35), "y": (-0.35, 0.35), "yaw": (-math.pi, math.pi)},
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

    reset_joints_by_scale = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.7, 1.3),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(                         # 重要修改 - 机器人随机推动事件，增强鲁棒性
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.0, 10.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5), 
                "y": (-0.5, 0.5), 
                "z": (-0.3, 0.3),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.3, 0.3),
            }
        },
    )

    # # interval
    # push_jump = EventTerm(                              # teacher - stage1 only
    #     func=mdp.push_when_still_stucked_random,
    #     mode="interval",
    #     interval_range_s=(1.0, 2.0),
    #     params={
    #         "command_name": "base_velocity",
    #         "vel_diff_threshold": 0.3,
    #         "stucked_counter_cnt": 3,
    #         "velocity_range": {
    #             "x": (0.75, 1.5), 
    #             "y": (0.0, 0.0), 
    #             "z": (0.75, 1.5),
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
    track_lin_vel_xy_exp0 = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(2.0)}
    )
    track_lin_vel_xy_exp1 = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.5)}
    )
    track_lin_vel_xy_exp2 = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # 角速度跟随奖励 [任务]
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # -- penalties
    # z 轴线速度惩罚 [姿态]
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # xy 轴角速度惩罚 [姿态]
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)
    # 姿态不水平惩罚 [姿态]
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    # 基座高度偏离惩罚 [姿态]
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-10.0,
        params={"target_height": 0.32, "sensor_cfg": SceneEntityCfg("base_height_scanner")},
    )
    # 默认站立姿态 [姿态]
    default_stand_pos = RewTerm(
        func=mdp.default_stand_pose,
        weight=-2.0,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    )
    # 站立速度惩罚 [能量]
    stand_still_vel = RewTerm(func=mdp.stand_still_vel, weight=-0.025, params={"command_name": "base_velocity"})
    # 关节力矩惩罚 [能量]
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.0003)
    # 关节加速度惩罚 [平滑]
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    # 基座速度突变惩罚 [平滑]
    base_acc_l2 = RewTerm(
        func=mdp.base_acc_l2,
        weight=-3.0e-6,
        params={"sensor_cfg": SceneEntityCfg("base_imu")},
    )
    # 动作频率惩罚 [平滑]
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_limit, weight=-0.01)
    # 关节软限制 [姿态]
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.2)
    # 前/后向运动时，对hip关节的软限制 [姿态]
    hip_pos_fb_limits = RewTerm(
        func=mdp.hip_joint_pos_fb_limits,
        weight=-5.0,
        params={"command_name": "base_velocity", "hip_limit_pos": -0.15, "asset_cfg": SceneEntityCfg("robot")},
    )
    # 左/右运动时，对hip关节的软限制 [姿态]
    hip_pos_lr_limits = RewTerm(
        func=mdp.hip_joint_pos_lr_limits,
        weight=-2.0,
        params={"command_name": "base_velocity", "hip_limit_pos": -0.30, "asset_cfg": SceneEntityCfg("robot")},
    )
    # 步态奖励
    trot_gait = RewTerm(
        func=mdp.trot_gait, 
        weight=1.0, 
        params={
            "command_name": "base_velocity", 
            "cycle_period": 0.7,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")
        }
    )
    # 抬腿高度奖励 [任务]
    feet_swing = RewTerm(
        func=mdp.feet_swing,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "target_height": 0.04,
            "cycle_period": 0.7, 
            "std": 0.4,
            "FL_foot_sensor_cfg": SceneEntityCfg("FL_foot_height_scanner"),
            "FR_foot_sensor_cfg": SceneEntityCfg("FR_foot_height_scanner"),
            "RL_foot_sensor_cfg": SceneEntityCfg("RL_foot_height_scanner"),
            "RR_foot_sensor_cfg": SceneEntityCfg("RR_foot_height_scanner"),
        }
    )
    # feet 滞空时间奖/惩 [姿态]
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.3,
        },
    )
    # feet slide
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.15,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        }
    )
    # feet 垂直面碰撞惩罚
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    # feet 同侧足端安全距离惩罚 [姿态]
    feet_spacing_exp = RewTerm(
        func=mdp.feet_spacing_exp,
        weight=-1.0,
        params={"safe_distance": 0.1, "std": 0.2},
    )
    # feet 同侧足碰撞惩罚 [姿态]
    feet_self_collision = RewTerm(
        func=mdp.feet_self_collision,
        weight=-50.0,
        params={"collision_distance": 0.05},
    )
    # feet 接触力惩罚
    feet_contact_force = RewTerm(
        func=mdp.contact_forces,
        weight=-0.08,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), "threshold": 100.0},
    )
    # 接触惩罚 [姿态]
    undesired_contacts_head = RewTerm(
        func=mdp.undesired_contacts,
        weight=-5.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="Head.*"), "threshold": 1.0},
    )
    # 接触惩罚 [姿态]
    undesired_contacts_thigh = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"), "threshold": 1.0},
    )
    # 接触惩罚 [姿态]
    undesired_contacts_calf = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"), "threshold": 1.0},
    )


# ============================================================================================================
# 定义终止条件 CFG


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
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
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**16

        # 修改传感器更新频率
        # 我们根据最小更新周期（物理更新周期）勾选所有传感器
        if self.scene.height_scanner is not None:  # 高度扫描仪
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt  # 50 Hz
        if self.scene.height_scanner_lf is not None:
            self.scene.height_scanner_lf.update_period = self.decimation * self.sim.dt  # 50 Hz
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

        # # 限定速度指令
        # self.commands.base_velocity.rel_vel_world_envs = 1.0
        # self.commands.base_velocity.ranges.lin_vel_x = (0.8, 1.2)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)

        # Play 播放时，机器人生成位置随机化，不跟随地形等级
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator.curriculum = False

        # 移除随机推力事件
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.add_base_mass.mass_distribution_params = (1.0, 3.0)

