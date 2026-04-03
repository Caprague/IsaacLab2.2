# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


# ============================================================================================================
# 导入资源


import os
import math

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, ImuCfg, patterns
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp

##
# Pre-defined configs - Rough Terrain
##
from omni.isaac.lab.terrains.config.rough_user import SKILL_RUN_TERRAINS_CFG  # isort: skip

##
# Pre-defined configs - Unitree Go2
##
from omni.isaac.lab_assets.unitree import UNITREE_GO2_CFG  # isort: skip

##
# GMS-defined local assets path
##
isaacsim_assets_path = os.environ.get("ISAACSIM_ASSETS_PATH")


# ============================================================================================================
# 定义交互场景 CFG


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # 地形
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=SKILL_RUN_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{isaacsim_assets_path}/Isaac/IsaacLab/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # 机器人
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 传感器(虚拟)
    # 高度扫描仪
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    # 接触力传感器
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # IMU
    base_imu = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/base", debug_vis=False)

    # 光源
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{isaacsim_assets_path}/Isaac/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# ============================================================================================================
# 定义命令 CFG


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 15.0),
        rel_standing_envs=0.05,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-2.0, 2.0), lin_vel_y=(-2.0, 2.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


# ============================================================================================================
# 定义动作 CFG


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


# ============================================================================================================
# 定义观测 CFG


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # 速度指令
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # 线速度
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        # 角速度
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        # 投影重力加速度
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        # 关节位置
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        # 关节速度
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        # 动作
        last_action = ObsTerm(func=mdp.last_action)
        # 高度扫描
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.5, 1.5),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


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
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
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

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_root_state_uniform = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.25, 0.25),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-1.0, 1.0),
            },
        },
    )

    reset_joints_by_scale = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = None  # 暂不启动
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(4.0, 9.0),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.3, 0.3)}},
    # )


# ============================================================================================================
# 定义奖励 CFG


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    # 线速度跟随奖励 [任务]
    track_lin_vel_xy_exp_easy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(4.0)}
    )
    track_lin_vel_xy_exp_hard = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.5)}
    )
    # 角速度跟随奖励 [任务]
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # -- penalties
    # z 轴线速度惩罚 [姿态]
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # xy 轴角速度惩罚 [姿态]
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # 姿态不水平惩罚 [姿态]
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)
    # 基座高度偏离惩罚 [姿态]
    base_height_l2 = RewTerm(
        func=mdp.base_height_to_feet_range_l1,
        weight=-5.0,
        params={"height_gap_target": 0.25, "diff_range": 0.05, "asset_cfg": SceneEntityCfg("robot", body_names="base")},
    )
    # 默认站立姿态 [姿态]
    default_stand_pos = RewTerm(
        func=mdp.default_stand_pose,
        weight=-1.0,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    )
    # 关节力矩惩罚 [能量]
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.0002)
    # 关节加速度惩罚 [平滑]
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.0e-7)
    # 基座速度突变惩罚 [平滑]
    base_acc_l2 = RewTerm(
        func=mdp.base_acc_l2,
        weight=-4.0e-6,
        params={"sensor_cfg": SceneEntityCfg("base_imu")},
    )
    # 动作频率惩罚 [平滑]
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # 关节软限制 [姿态]
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.03)
    # 前/后向运动时，对hip关节的软限制 [姿态]
    hip_pos_limits = RewTerm(
        func=mdp.hip_joint_pos_limits,
        weight=-1.5,
        params={"command_name": "base_velocity", "hip_limit_pos": -0.15, "asset_cfg": SceneEntityCfg("robot")},
    )
    # feet 滞空时间奖/惩 [姿态]
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    # 接触惩罚 [姿态]
    undesired_contacts_head = RewTerm(
        func=mdp.undesired_contacts,
        weight=-10.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="Head.*"), "threshold": 1.0},
    )
    # 接触惩罚 [姿态]
    undesired_contacts_thigh = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"), "threshold": 1.0},
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

        # 修改传感器更新频率
        # 我们根据最小更新周期（物理更新周期）勾选所有传感器
        if self.scene.height_scanner is not None:  # 高度扫描仪
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt  # 50 Hz
        if self.scene.contact_forces is not None:  # 接触力传感器
            self.scene.contact_forces.update_period = self.sim.dt  # 200 Hz
        if self.scene.base_imu is not None:  # IMU
            self.scene.base_imu.update_period = self.sim.dt  # 200 Hz

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

        # 限定速度指令
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.resampling_time_range = (10, 10)
        self.terminations.time_out.time_out = 30

        # Play 播放时，机器人生成位置随机化，不跟随地形等级
        self.scene.terrain.max_init_terrain_level = None

        # Play 播放时禁用策略随机性
        self.observations.policy.enable_corruption = False

        # 移除随机推力事件
        self.events.base_external_force_torque = None
        self.events.push_robot = None
