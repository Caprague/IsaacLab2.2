# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


# ----------------------------------------------------------------------------------------------------------------------------------
def feet_state_std(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the feet state(time)
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    #  compute the std
    air_time_std = torch.std(last_air_time, dim=-1, unbiased=True)
    contact_time_std = torch.std(last_contact_time, dim=-1, unbiased=True)
    # no reward for zero command
    reward = (air_time_std + contact_time_std) * torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_stumble(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg, 
) -> torch.Tensor:
    """
        Penalize feet hitting vertical surfaces by checking contact forces history.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    feet_contact_forces = contact_sensor.data.net_forces_w_history
    return torch.sum(feet_contact_forces[:, :, sensor_cfg.body_ids, :2].norm(dim=-1) > \
          5 * feet_contact_forces[:, :, sensor_cfg.body_ids, 2].abs(), dim=-1).sum(dim=-1)


def base_acc_l2(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    imu_sensor: Imu = env.scene.sensors[sensor_cfg.name]
    # compute the error
    lin_acc_b = imu_sensor.data.lin_acc_b
    ang_acc_b = imu_sensor.data.ang_acc_b
    return torch.sum(torch.square(lin_acc_b) + torch.square(ang_acc_b), dim=-1)


def hip_joint_pos_fb_limits(    # 前后运动时，hip 关节位置超出限制的惩罚
    env: ManagerBasedRLEnv,
    command_name: str,
    hip_limit_pos: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    vel_command_xy = env.command_manager.get_command(command_name)[:, :2]
    mask = torch.abs(vel_command_xy[:, 0]) > (4 * torch.abs(vel_command_xy[:, 1]))
    # compute out of limits constraints
    l_hip_joint_idx, _ = asset.find_joints(".*L_hip_joint")  # 默认 0.1, 倾向于 > 0
    r_hip_joint_idx, _ = asset.find_joints(".*R_hip_joint")  # 默认 -0.1, 倾向于 < 0
    l_hip_joint_pos = asset.data.joint_pos[:, l_hip_joint_idx]
    r_hip_joint_pos = asset.data.joint_pos[:, r_hip_joint_idx]
    l_out_of_limit = (hip_limit_pos * torch.ones_like(l_hip_joint_pos) - l_hip_joint_pos).clip(min=0.0)
    r_out_of_limit = (r_hip_joint_pos + hip_limit_pos * torch.ones_like(r_hip_joint_pos)).clip(min=0.0)

    # print("l_hip_joint_pos: ", l_hip_joint_pos)
    # print("r_hip_joint_pos: ", r_hip_joint_pos)

    return torch.sum(l_out_of_limit + r_out_of_limit, dim=1) * mask


def hip_joint_pos_lr_limits(    # 左右运动时，hip 关节位置超出限制的惩罚
    env: ManagerBasedRLEnv,
    command_name: str,
    hip_limit_pos: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    vel_command_xy = env.command_manager.get_command(command_name)[:, :2]
    mask = torch.abs(vel_command_xy[:, 1]) > (4 * torch.abs(vel_command_xy[:, 0]))
    # compute out of limits constraints
    l_hip_joint_idx, _ = asset.find_joints(".*L_hip_joint")  # 默认 0.1, 倾向于 > 0
    r_hip_joint_idx, _ = asset.find_joints(".*R_hip_joint")  # 默认 -0.1, 倾向于 < 0
    l_hip_joint_pos = asset.data.joint_pos[:, l_hip_joint_idx]
    r_hip_joint_pos = asset.data.joint_pos[:, r_hip_joint_idx]
    l_out_of_limit = (hip_limit_pos * torch.ones_like(l_hip_joint_pos) - l_hip_joint_pos).clip(min=0.0)
    r_out_of_limit = (r_hip_joint_pos + hip_limit_pos * torch.ones_like(r_hip_joint_pos)).clip(min=0.0)
    return torch.sum(l_out_of_limit + r_out_of_limit, dim=1) * mask


# 惩罚 - 指令静止站立时，与默认姿态间的偏差
def default_stand_pose(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    mask = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < 0.1
    return torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1) * mask


def stand_still_vel(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """指令静止站立时, base 速度及 joint 速度非零的惩罚"""
    asset: RigidObject = env.scene[asset_cfg.name]
    stand_mask = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < 0.1
    base_lin_vel = torch.norm(asset.data.root_lin_vel_b, dim=1)
    joint_vel = torch.sum(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    penalty = (base_lin_vel * 50.0 + joint_vel) * stand_mask
    # print(f"stand_mask: {stand_mask}")
    # print(f"base_lin_vel: {base_lin_vel}")
    # print(f"joint_vel: {joint_vel}")
    return penalty.clamp(max=100.0)


# 惩罚 - 足端平均高度到 base 高度差不足 [定点高度版] (Go2限定，因直接读取 body_pos_w 数据)
def base_height_to_feet_l1(
    env: ManagerBasedRLEnv,
    height_gap_target: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """足端到 base 距离不足惩罚"""
    # 获取资产对象（这里假设env.scene已经能够直接通过名字索引到RigidObject）
    asset: RigidObject = env.scene[asset_cfg.name]
    # 获取根的高度 (base)
    base_height = asset.data.root_pos_w[:, 2]
    # 获取足端高度 (各机器人的四个foot)
    feet_height = asset.data.body_pos_w[:, -4:, 2]
    # 对单个机器人的四个足端高度取平均
    feet_height_ave = feet_height.mean(dim=1)
    # 计算 base 到足端平均高度的高度差，并直接与目标高度差求差的绝对值作为惩罚
    return torch.abs(base_height - feet_height_ave - height_gap_target)


# 惩罚 - 足端平均高度到 base 高度差不足 [容差高度版] (Go2限定，因直接读取 body_pos_w 数据)
def base_height_to_feet_range_l1(
    env: ManagerBasedRLEnv,
    height_gap_target: float,
    diff_range: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """足端到 base 距离不足惩罚"""
    # 获取资产对象（这里假设env.scene已经能够直接通过名字索引到RigidObject）
    asset: RigidObject = env.scene[asset_cfg.name]
    # 获取根的高度 (base)
    base_height = asset.data.root_pos_w[:, 2]
    # 获取足端高度 (各机器人的四个foot)
    feet_height = asset.data.body_pos_w[:, -4:, 2]
    # 对单个机器人的四个足端高度取平均
    feet_height_ave = feet_height.mean(dim=1)
    # 计算 base 到足端平均高度的高度差的绝对值
    height_diff = torch.abs(torch.abs(base_height - feet_height_ave) - height_gap_target * torch.ones_like(base_height))
    # 若高度差与目标高度差小于容限，则返 0 ；反之返回惩罚值(正值)
    reward = height_diff - diff_range * torch.ones_like(height_diff)
    reward *= reward > 0

    # # Debug
    # print(f"base_height : {base_height}")
    # print(f"feet_height : {feet_height}")
    # print(f"feet_height_ave : {feet_height_ave}")
    # print(f"height_diff : {height_diff}")
    # print(f"reward : {reward}")
    # print("===============================================")

    return reward


# 疑点： 单纯奖励抬腿高度，是否会导致不自然的动作偏好？
def feet_lift_height_to_base(
    env: ManagerBasedRLEnv,
    height_threshold: float,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    flat_threshold: float = 0.1
) -> torch.Tensor:
    """Reward for lifting feet above threshold when base is flat.
    
    Args:
        env: The environment instance.
        height_threshold: Minimum height difference between feet and base to reward.
        command_name: Name of the command used to control the robot.
        sensor_cfg: Configuration for the feet contact sensors.
        asset_cfg: Configuration for the robot asset.
        flat_threshold: Maximum allowed base orientation deviation from flat.
    """
    # Check if base is flat enough
    asset: RigidObject = env.scene[asset_cfg.name]
    is_flat = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1) < flat_threshold
    # Check if command is active (similar to default_stand_pose)
    is_command_active = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1

    # Get feet positions relative to base
    feet_height = asset.data.body_pos_w[:, -4:, 2]
    base_height = asset.data.root_pos_w[:, 2]
    height_diff = base_height.unsqueeze(1) - feet_height
    foot_lifted = torch.any(height_diff < height_threshold, dim=1)
    
    # Compute reward (only when base is flat and feet are lifted)
    reward = torch.where(
        is_flat & is_command_active & foot_lifted,
        torch.ones_like(base_height),
        torch.zeros_like(base_height)
    )
    return reward


def feet_swing(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_height: float,
    cycle_period: float,
    FL_foot_sensor_cfg: SceneEntityCfg,
    FR_foot_sensor_cfg: SceneEntityCfg,
    RL_foot_sensor_cfg: SceneEntityCfg,
    RR_foot_sensor_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    """
        Reward for lifting feet above threshold when base is flat.
    """
    # Check if command is active (similar to default_stand_pose)
    is_command_active = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1   # N

    # ground height
    FL_foot_sensor: RayCaster = env.scene.sensors[FL_foot_sensor_cfg.name]
    FR_foot_sensor: RayCaster = env.scene.sensors[FR_foot_sensor_cfg.name]
    RL_foot_sensor: RayCaster = env.scene.sensors[RL_foot_sensor_cfg.name]
    RR_foot_sensor: RayCaster = env.scene.sensors[RR_foot_sensor_cfg.name]
    FL_foot_height = FL_foot_sensor.data.pos_w[:, 2].unsqueeze(1) - FL_foot_sensor.data.ray_hits_w[..., 2]    # N x 1
    FR_foot_height = FR_foot_sensor.data.pos_w[:, 2].unsqueeze(1) - FR_foot_sensor.data.ray_hits_w[..., 2]
    RL_foot_height = RL_foot_sensor.data.pos_w[:, 2].unsqueeze(1) - RL_foot_sensor.data.ray_hits_w[..., 2]
    RR_foot_height = RR_foot_sensor.data.pos_w[:, 2].unsqueeze(1) - RR_foot_sensor.data.ray_hits_w[..., 2]

    # print(f"FL_foot_height : {FL_foot_height}")
    # print(f"FR_foot_height : {FR_foot_height}")
    # print(f"RL_foot_height : {RL_foot_height}")
    # print(f"RR_foot_height : {RR_foot_height}")
    
    # mask 1 is stance, 0 is swing
    phase = (env.episode_length_buf * env.step_dt) % cycle_period / cycle_period
    stance_mask = torch.zeros((env.num_envs, 2), device=env.device)
    stance_mask[:, 0] = (phase < 0.5).float()
    stance_mask[:, 1] = (phase > 0.5).float()
    swing_mask = 1 - stance_mask    # N x 2

    # compute feet union error
    foot_target_height = (torch.sin(2*torch.pi*phase) * target_height).abs().unsqueeze(1)
    feet_union_error_FL_RR = ((FL_foot_height - foot_target_height).abs() + (RR_foot_height - foot_target_height).abs())*(swing_mask[:, 0].unsqueeze(1))
    feet_union_error_FR_RL = ((FR_foot_height - foot_target_height).abs() + (RL_foot_height - foot_target_height).abs())*(swing_mask[:, 1].unsqueeze(1))

    reward1 = torch.exp(-feet_union_error_FL_RR/ std**2)*(swing_mask[:, 0].unsqueeze(1))
    reward2 = torch.exp(-feet_union_error_FR_RL/ std**2)*(swing_mask[:, 1].unsqueeze(1))

    # print(f"FL_foot_height : {FL_foot_height}")
    # print(f"FR_foot_height : {FR_foot_height}")
    # print(f"RL_foot_height : {RL_foot_height}")
    # print(f"RR_foot_height : {RR_foot_height}")
    # print(f"swing_mask : {swing_mask}")
    # print(f"foot_target_height : {foot_target_height}")
    # print(f"reward1 : {reward1}")
    # print(f"reward2 : {reward2}")

    return (reward1 * 0.5 + reward2 * 0.5).view(-1) * is_command_active


def feet_swing_vel(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_height: float,
    cycle_period: tuple[torch.Tensor, torch.Tensor],
    velocity_limit: tuple[torch.Tensor, torch.Tensor],
    FL_foot_sensor_cfg: SceneEntityCfg,
    FR_foot_sensor_cfg: SceneEntityCfg,
    RL_foot_sensor_cfg: SceneEntityCfg,
    RR_foot_sensor_cfg: SceneEntityCfg,
    std: float,
    binary_threshold: float = 0.5,
) -> torch.Tensor:
    """
        Reward for lifting feet above threshold when base is flat.
    """
    # Check if command is active (similar to default_stand_pose)
    vel_cmd = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    is_command_active = vel_cmd > 0.1   # N

    # ground height
    FL_foot_sensor: RayCaster = env.scene.sensors[FL_foot_sensor_cfg.name]
    FR_foot_sensor: RayCaster = env.scene.sensors[FR_foot_sensor_cfg.name]
    RL_foot_sensor: RayCaster = env.scene.sensors[RL_foot_sensor_cfg.name]
    RR_foot_sensor: RayCaster = env.scene.sensors[RR_foot_sensor_cfg.name]
    FL_foot_height = FL_foot_sensor.data.pos_w[:, 2].unsqueeze(1) - FL_foot_sensor.data.ray_hits_w[..., 2]    # N x 1
    FR_foot_height = FR_foot_sensor.data.pos_w[:, 2].unsqueeze(1) - FR_foot_sensor.data.ray_hits_w[..., 2]
    RL_foot_height = RL_foot_sensor.data.pos_w[:, 2].unsqueeze(1) - RL_foot_sensor.data.ray_hits_w[..., 2]
    RR_foot_height = RR_foot_sensor.data.pos_w[:, 2].unsqueeze(1) - RR_foot_sensor.data.ray_hits_w[..., 2]

    # print(f"FL_foot_height : {FL_foot_height}")
    # print(f"FR_foot_height : {FR_foot_height}")
    # print(f"RL_foot_height : {RL_foot_height}")
    # print(f"RR_foot_height : {RR_foot_height}")
    
    # mask 1 is stance, 0 is swing
    vel_cmd = vel_cmd.clamp(min=velocity_limit[0], max=velocity_limit[1])
    cyc_val = cycle_period[1]*torch.ones_like(vel_cmd) - (vel_cmd - velocity_limit[0])*(cycle_period[1] - cycle_period[0])/(velocity_limit[1] - velocity_limit[0])
    phase = (env.episode_length_buf * env.step_dt) % cyc_val / cyc_val
    sin_phase_abs = torch.sin(2 * torch.pi * phase).abs()
    cos_phase_abs = torch.cos(2 * torch.pi * phase).abs()
    stance_mask = torch.zeros((env.num_envs, 2), device=env.device)
    stance_mask[:, 0] = (cos_phase_abs > binary_threshold).float()
    stance_mask[:, 1] = (sin_phase_abs > binary_threshold).float()
    swing_mask = 1 - stance_mask    # N x 2

    # compute feet union error
    foot_target_height_FL_RR = (cos_phase_abs*target_height).unsqueeze(1)
    foot_target_height_FR_RL = (sin_phase_abs*target_height).unsqueeze(1)
    feet_union_error_FL_RR = ((FL_foot_height - foot_target_height_FL_RR).abs() + (RR_foot_height - foot_target_height_FL_RR).abs())*(swing_mask[:, 0].unsqueeze(1))
    feet_union_error_FR_RL = ((FR_foot_height - foot_target_height_FR_RL).abs() + (RL_foot_height - foot_target_height_FR_RL).abs())*(swing_mask[:, 1].unsqueeze(1))

    reward1 = torch.exp(-feet_union_error_FL_RR/ std**2)*(swing_mask[:, 0].unsqueeze(1))
    reward2 = torch.exp(-feet_union_error_FR_RL/ std**2)*(swing_mask[:, 1].unsqueeze(1))

    # print(f"FL_foot_height : {FL_foot_height}")
    # print(f"FR_foot_height : {FR_foot_height}")
    # print(f"RL_foot_height : {RL_foot_height}")
    # print(f"RR_foot_height : {RR_foot_height}")
    # print(f"swing_mask : {swing_mask}")
    # print(f"foot_target_height : {foot_target_height}")
    # print(f"reward1 : {reward1}")
    # print(f"reward2 : {reward2}")

    return (reward1 * 0.5 + reward2 * 0.5).view(-1) * is_command_active


def feet_swing_vel_v2(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_height: tuple[torch.Tensor, torch.Tensor],
    cycle_period: tuple[torch.Tensor, torch.Tensor],
    velocity_limit: tuple[torch.Tensor, torch.Tensor],
    FL_foot_sensor_cfg: SceneEntityCfg,
    FR_foot_sensor_cfg: SceneEntityCfg,
    RL_foot_sensor_cfg: SceneEntityCfg,
    RR_foot_sensor_cfg: SceneEntityCfg,
    std: float,
    binary_threshold: float = 0.5,
) -> torch.Tensor:
    """
        Reward for lifting feet above threshold when base is flat.
    """
    # Check if command is active (similar to default_stand_pose)
    vel_cmd = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    is_command_active = vel_cmd > 0.1   # N

    # 速度归一化处理
    vel_normalized = (vel_cmd - velocity_limit[0]) / (velocity_limit[1] - velocity_limit[0])
    vel_normalized = vel_normalized.clamp(0.0, 1.0)
    # 线性插值计算动态高度
    dynamic_target_height = target_height[0] + vel_normalized * (target_height[1] - target_height[0])

    # ground height
    FL_foot_sensor: RayCaster = env.scene.sensors[FL_foot_sensor_cfg.name]
    FR_foot_sensor: RayCaster = env.scene.sensors[FR_foot_sensor_cfg.name]
    RL_foot_sensor: RayCaster = env.scene.sensors[RL_foot_sensor_cfg.name]
    RR_foot_sensor: RayCaster = env.scene.sensors[RR_foot_sensor_cfg.name]
    FL_foot_height = FL_foot_sensor.data.pos_w[:, 2].unsqueeze(1) - FL_foot_sensor.data.ray_hits_w[..., 2]    # N x 1
    FR_foot_height = FR_foot_sensor.data.pos_w[:, 2].unsqueeze(1) - FR_foot_sensor.data.ray_hits_w[..., 2]
    RL_foot_height = RL_foot_sensor.data.pos_w[:, 2].unsqueeze(1) - RL_foot_sensor.data.ray_hits_w[..., 2]
    RR_foot_height = RR_foot_sensor.data.pos_w[:, 2].unsqueeze(1) - RR_foot_sensor.data.ray_hits_w[..., 2]

    # print(f"FL_foot_height : {FL_foot_height}")
    # print(f"FR_foot_height : {FR_foot_height}")
    # print(f"RL_foot_height : {RL_foot_height}")
    # print(f"RR_foot_height : {RR_foot_height}")
    
    # mask 1 is stance, 0 is swing
    vel_cmd = vel_cmd.clamp(min=velocity_limit[0], max=velocity_limit[1])
    cyc_val = cycle_period[1]*torch.ones_like(vel_cmd) - (vel_cmd - velocity_limit[0])*(cycle_period[1] - cycle_period[0])/(velocity_limit[1] - velocity_limit[0])
    phase = (env.episode_length_buf * env.step_dt) % cyc_val / cyc_val
    sin_phase_abs = torch.sin(2 * torch.pi * phase).abs()
    cos_phase_abs = torch.cos(2 * torch.pi * phase).abs()
    stance_mask = torch.zeros((env.num_envs, 2), device=env.device)
    stance_mask[:, 0] = (cos_phase_abs > binary_threshold).float()
    stance_mask[:, 1] = (sin_phase_abs > binary_threshold).float()
    swing_mask = 1 - stance_mask    # N x 2

    # compute feet union error
    foot_target_height_FL_RR = (cos_phase_abs*dynamic_target_height).unsqueeze(1)
    foot_target_height_FR_RL = (sin_phase_abs*dynamic_target_height).unsqueeze(1)
    feet_union_error_FL_RR = ((FL_foot_height - foot_target_height_FL_RR).abs() + (RR_foot_height - foot_target_height_FL_RR).abs())*(swing_mask[:, 0].unsqueeze(1))
    feet_union_error_FR_RL = ((FR_foot_height - foot_target_height_FR_RL).abs() + (RL_foot_height - foot_target_height_FR_RL).abs())*(swing_mask[:, 1].unsqueeze(1))

    reward1 = torch.exp(-feet_union_error_FL_RR/ std**2)*(swing_mask[:, 0].unsqueeze(1))
    reward2 = torch.exp(-feet_union_error_FR_RL/ std**2)*(swing_mask[:, 1].unsqueeze(1))

    # print(f"FL_foot_height : {FL_foot_height}")
    # print(f"FR_foot_height : {FR_foot_height}")
    # print(f"RL_foot_height : {RL_foot_height}")
    # print(f"RR_foot_height : {RR_foot_height}")
    # print(f"swing_mask : {swing_mask}")
    # print(f"dynamic_target_height : {dynamic_target_height}")
    # print(f"reward1 : {reward1}")
    # print(f"reward2 : {reward2}")

    return (reward1 * 0.5 + reward2 * 0.5).view(-1) * is_command_active


def trot_gait(
    env: ManagerBasedRLEnv,
    command_name: str,
    cycle_period: float,
    sensor_cfg: SceneEntityCfg, 
    threshold: float = 5.0,
) -> torch.Tensor:
    """
    Calculates a reward based on the number of feet contacts aligning with the gait phase. 
    Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
    """
    # Check if command is active
    is_command_active = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1   # N

    # Get contact sensor states
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_mask = (net_contact_forces[:, 0, sensor_cfg.body_ids, 2] > threshold).float()

    # Get phase
    phase = (env.episode_length_buf * env.step_dt) % cycle_period / cycle_period
    stance_mask = torch.zeros((env.num_envs, 2), device=env.device)
    stance_mask[:, 0] = (phase < 0.5).float()
    stance_mask[:, 1] = (phase > 0.5).float()

    # compute reward
    reward_mask = (contact_mask[:, 0] == contact_mask[:, 3]) & \
                  (contact_mask[:, 1] == contact_mask[:, 2]) & \
                  (contact_mask[:, 0] == stance_mask[:, 0]) & \
                  (contact_mask[:, 1] == stance_mask[:, 1])

    # print(f"contact_forces : {net_contact_forces[:, 0, sensor_cfg.body_ids, 2]}")
    # print(f"contact_mask : {contact_mask}")
    # print(f"stance_mask : {stance_mask}")
    # print(f"reward_mask : {reward_mask}")

    return reward_mask.float() * is_command_active


def trot_gait_vel(
    env: ManagerBasedRLEnv,
    command_name: str,
    cycle_period: tuple[torch.Tensor, torch.Tensor],
    velocity_limit: tuple[torch.Tensor, torch.Tensor],
    sensor_cfg: SceneEntityCfg, 
    threshold: float = 5.0,
    binary_threshold: float = 0.5,
) -> torch.Tensor:
    """
    Calculates a reward based on the number of feet contacts aligning with the gait phase. 
    Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
    """
    # Check if command is active
    vel_cmd = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    is_command_active = vel_cmd > 0.1   # N

    # Get contact sensor states
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_mask = (net_contact_forces[:, 0, sensor_cfg.body_ids, 2] > threshold).float()

    # Get phase
    vel_cmd = vel_cmd.clamp(min=velocity_limit[0], max=velocity_limit[1])
    cyc_val = cycle_period[1]*torch.ones_like(vel_cmd) - (vel_cmd - velocity_limit[0])*(cycle_period[1] - cycle_period[0])/(velocity_limit[1] - velocity_limit[0])
    phase = (env.episode_length_buf * env.step_dt) % cyc_val / cyc_val
    sin_phase_abs = torch.sin(2 * torch.pi * phase).abs()
    cos_phase_abs = torch.cos(2 * torch.pi * phase).abs()
    stance_mask = torch.zeros((env.num_envs, 2), device=env.device)
    stance_mask[:, 0] = (cos_phase_abs > binary_threshold).float()
    stance_mask[:, 1] = (sin_phase_abs > binary_threshold).float()

    # compute reward
    reward_mask = (contact_mask[:, 0] == contact_mask[:, 3]) & \
                  (contact_mask[:, 1] == contact_mask[:, 2]) & \
                  (contact_mask[:, 0] == stance_mask[:, 0]) & \
                  (contact_mask[:, 1] == stance_mask[:, 1])

    # print(f"contact_forces : {net_contact_forces[:, 0, sensor_cfg.body_ids, 2]}")
    # print(f"contact_mask : {contact_mask}")
    # print(f"stance_mask : {stance_mask}")
    # print(f"reward_mask : {reward_mask}")

    return reward_mask.float() * is_command_active


def gallop_gait(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg, 
    threshold: float = 5.0,
) -> torch.Tensor:
    """
    Calculates a reward based on the number of feet contacts aligning with the gait phase. 
    Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
    """
    # Check if command is active
    is_command_active = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1   # N

    # Get contact sensor states
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_mask = (net_contact_forces[:, 0, sensor_cfg.body_ids, 2] > threshold)

    # gallop reward
    gallop_reward = (contact_mask[:, 0] == contact_mask[:, 1]) & \
                    (contact_mask[:, 2] == contact_mask[:, 3]) & \
                    (contact_mask[:, 0] != contact_mask[:, 2])

    return gallop_reward.float() * is_command_active


def gallop_gait_v2(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 5.0,
    sync_std: float = 0.6,
    target_contact_duration: float = 0.3,
    duration_std: float = 0.6,
) -> torch.Tensor:
    """
    Calculates a reward combining gallop gait pattern with contact time synchronization.
    
    This function rewards the agent for:
    1. Maintaining gallop gait pattern: diagonal feet contact together, and front/rear diagonals alternate
    2. Encouraging sufficient contact time for each diagonal pair (using smooth discount)
    3. Synchronizing contact timing between diagonal feet (0-2 and 1-3 pairs)
    
    Args:
        env: The environment instance.
        command_name: Name of the velocity command.
        sensor_cfg: Configuration for the contact sensor.
        force_threshold: Force threshold to determine foot contact.
        sync_std: Standard deviation for Gaussian decay of contact time synchronization between diagonal feet.
        duration_std: Standard deviation for Gaussian decay of contact time duration.
        target_contact_duration: Target average contact duration for efficient contact (used as reference for discount).
    
    Returns:
        Reward tensor combining gait pattern and contact time synchronization.
    """
    # Check if command is active
    is_command_active = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1  # N

    # Get contact sensor states
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # condition 1: gallop gait
    # Get current contact mask
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_mask = (net_contact_forces[:, 0, sensor_cfg.body_ids, 2] > force_threshold)
    # Gallop gait pattern reward: diagonal feet contact together, front/rear diagonals alternate
    gallop_pattern = (contact_mask[:, 0] == contact_mask[:, 1]) & \
                     (contact_mask[:, 2] == contact_mask[:, 3]) & \
                     (contact_mask[:, 0] != contact_mask[:, 2])

    # condition 2: contact time duration discount
    # Get last contact time for each foot
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    ave_contact_duration_01 = (last_contact_time[:, 0] + last_contact_time[:, 1]) * 0.5
    ave_contact_duration_23 = (last_contact_time[:, 2] + last_contact_time[:, 3]) * 0.5
    ave_contact_duration_all = (ave_contact_duration_01 + ave_contact_duration_23) * 0.5
    # Use Gaussian decay: when ave_contact_duration_all >= target_contact_duration, discount = 1.0
    # When ave_contact_duration_all < target_contact_duration, discount decays based on difference
    contact_duration_discount = torch.exp(-torch.clamp(target_contact_duration - ave_contact_duration_all, min=0.0) / (duration_std)**2)

    # condition 3: contact time synchronization discount
    contact_time_diff = torch.abs(ave_contact_duration_01 - ave_contact_duration_23)
    sync_discount = torch.exp(-contact_time_diff/(sync_std)**2)

    # Combine rewards: gait pattern (0 or 1) scaled by time synchronization discount
    combined_reward = gallop_pattern.float() * sync_discount * contact_duration_discount

    return combined_reward * is_command_active


def feet_spacing_exp(
    env: ManagerBasedRLEnv,
    safe_distance: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """计算左右同侧足与前后同侧足间的距离惩罚"""
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # 获取四个足端位置 (假设顺序为: 左前, 右前, 左后, 右后)
    feet_pos = asset.data.body_pos_w[:, -4:, :]
    
    # 计算左右同侧足距离 (左前-左后, 右前-右后)
    left_pair_dist = torch.norm(feet_pos[:, 0] - feet_pos[:, 2], dim=1)
    right_pair_dist = torch.norm(feet_pos[:, 1] - feet_pos[:, 3], dim=1)
    
    # 计算前后同侧足距离 (左前-右前, 左后-右后)
    front_pair_dist = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=1)
    rear_pair_dist = torch.norm(feet_pos[:, 2] - feet_pos[:, 3], dim=1)
    
    # 计算所有配对的最小距离
    min_dist = torch.stack([left_pair_dist, right_pair_dist, front_pair_dist, rear_pair_dist]).min(dim=0)[0]

    # 当最小距离小于安全距离时给予惩罚
    mask = (safe_distance - min_dist) > 0
    penalty = min_dist * mask  # 仅当距离小于安全距离时保留正值
    penalty = torch.exp(-penalty/(std)**2) * mask   # 安全距离足够的不予惩罚
    
    # print(f"min_dist : {min_dist}")
    # print(f"penalty : {penalty}")
    # if torch.any(mask):
    #     print(f"min_dist : {min_dist[mask]}")
    
    return penalty


def feet_gap_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """计算左右同侧足与前后同侧足间的距离惩罚"""
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # 获取四个足端位置 (假设顺序为: 左前, 右前, 左后, 右后)
    feet_pos = asset.data.body_pos_w[:, -4:, :]
    
    # 计算左右同侧足距离 (左前-左后, 右前-右后)
    left_pair_dist = torch.norm(feet_pos[:, 0] - feet_pos[:, 2], dim=1)
    right_pair_dist = torch.norm(feet_pos[:, 1] - feet_pos[:, 3], dim=1)
    
    # 计算前后同侧足距离 (左前-右前, 左后-右后)
    front_pair_dist = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=1)
    rear_pair_dist = torch.norm(feet_pos[:, 2] - feet_pos[:, 3], dim=1)
    
    # 计算所有配对的最小距离
    min_dist = torch.stack([left_pair_dist, right_pair_dist, front_pair_dist, rear_pair_dist]).min(dim=0)[0]
    penalty = torch.exp(-min_dist/(std)**2)
    
    # print(f"min_dist : {min_dist}")
    # print(f"penalty : {penalty}")
    # if torch.any(mask):
    #     print(f"min_dist : {min_dist[mask]}")
    
    return penalty


def feet_self_collision(
    env: ManagerBasedRLEnv,
    collision_distance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """计算左右同侧足与前后同侧足间的距离惩罚"""
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # 获取四个足端位置 (假设顺序为: 左前, 右前, 左后, 右后)
    feet_pos = asset.data.body_pos_w[:, -4:, :]
    
    # 计算左右同侧足距离 (左前-左后, 右前-右后)
    left_pair_dist = torch.norm(feet_pos[:, 0] - feet_pos[:, 2], dim=1)
    right_pair_dist = torch.norm(feet_pos[:, 1] - feet_pos[:, 3], dim=1)
    
    # 计算前后同侧足距离 (左前-右前, 左后-右后)
    front_pair_dist = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=1)
    rear_pair_dist = torch.norm(feet_pos[:, 2] - feet_pos[:, 3], dim=1)
    
    # 计算所有配对的最小距离
    min_dist = torch.minimum(
        torch.minimum(left_pair_dist, right_pair_dist),
        torch.minimum(front_pair_dist, rear_pair_dist)
    )

    # 当最小距离小于碰撞距离时给予惩罚
    mask = (collision_distance - min_dist) > 0
    penalty = mask.float()
    
    # print(f"min_dist : {min_dist}")
    # print(f"penalty : {penalty}")
    # if torch.any(mask):
    #     print(f"min_dist : {min_dist[mask]}")
    
    return penalty


def feet_last_air_height(
    env: ManagerBasedRLEnv,
    # contact sensor -> contact mask
    sensor_cfg: SceneEntityCfg, 
    contact_threshold: float,
    # velocity command -> should move
    command_name: str,
    # feet raycast sensor -> feet air height
    FL_foot_sensor_cfg: SceneEntityCfg,
    FR_foot_sensor_cfg: SceneEntityCfg,
    RL_foot_sensor_cfg: SceneEntityCfg,
    RR_foot_sensor_cfg: SceneEntityCfg,
    # target minimum height
    height_threshold: float,
) -> torch.Tensor:
    if not hasattr(env, "feet_last_air_height"):
        env.feet_last_air_height = torch.zeros((env.num_envs, 4), device=env.device)

    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the contact mask
    first_contact_mask = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]

    # Check if command is active (similar to default_stand_pose)
    vel_cmd = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    is_command_active = vel_cmd > 0.1   # N

    # fett air height
    FL_foot_sensor: RayCaster = env.scene.sensors[FL_foot_sensor_cfg.name]
    FR_foot_sensor: RayCaster = env.scene.sensors[FR_foot_sensor_cfg.name]
    RL_foot_sensor: RayCaster = env.scene.sensors[RL_foot_sensor_cfg.name]
    RR_foot_sensor: RayCaster = env.scene.sensors[RR_foot_sensor_cfg.name]
    FL_foot_height = FL_foot_sensor.data.pos_w[:, 2].unsqueeze(1) - FL_foot_sensor.data.ray_hits_w[..., 2]    # N x 1
    FR_foot_height = FR_foot_sensor.data.pos_w[:, 2].unsqueeze(1) - FR_foot_sensor.data.ray_hits_w[..., 2]
    RL_foot_height = RL_foot_sensor.data.pos_w[:, 2].unsqueeze(1) - RL_foot_sensor.data.ray_hits_w[..., 2]
    RR_foot_height = RR_foot_sensor.data.pos_w[:, 2].unsqueeze(1) - RR_foot_sensor.data.ray_hits_w[..., 2]
    env.feet_last_air_height[:, 0] = torch.where(env.feet_last_air_height[:, 0].unsqueeze(1) < FL_foot_height, FL_foot_height, env.feet_last_air_height[:, 0].unsqueeze(1)).squeeze(-1)
    env.feet_last_air_height[:, 1] = torch.where(env.feet_last_air_height[:, 1].unsqueeze(1) < FR_foot_height, FR_foot_height, env.feet_last_air_height[:, 1].unsqueeze(1)).squeeze(-1)
    env.feet_last_air_height[:, 2] = torch.where(env.feet_last_air_height[:, 2].unsqueeze(1) < RL_foot_height, RL_foot_height, env.feet_last_air_height[:, 2].unsqueeze(1)).squeeze(-1)
    env.feet_last_air_height[:, 3] = torch.where(env.feet_last_air_height[:, 3].unsqueeze(1) < RR_foot_height, RR_foot_height, env.feet_last_air_height[:, 3].unsqueeze(1)).squeeze(-1)

    # is feet_contact_mask true then return reward value(related to feet_last_air_height)
    reward = torch.sum((height_threshold - env.feet_last_air_height).clamp_min(0.0) * first_contact_mask, dim=-1) * is_command_active
    env.feet_last_air_height = torch.where(first_contact_mask, torch.zeros_like(env.feet_last_air_height, device=env.device), env.feet_last_air_height)
    return reward


def gait_v1(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg, 
    threshold: float = 5.0,
) -> torch.Tensor:
    """
    Calculates a reward based on the number of feet contacts aligning with the gait phase. 
    Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
    """
    # Check if command is active
    is_command_active = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1   # N

    # Get contact sensor states
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_mask = (net_contact_forces[:, 0, sensor_cfg.body_ids, 2] > threshold)

    # trot gait
    trot_reward = (contact_mask[:, 0] == contact_mask[:, 3]) & \
                  (contact_mask[:, 1] == contact_mask[:, 2]) & \
                  (contact_mask[:, 0] != contact_mask[:, 1])
    
    # gallop reward
    gallop_reward = (contact_mask[:, 0] == contact_mask[:, 1]) & \
                    (contact_mask[:, 2] == contact_mask[:, 3]) & \
                    (contact_mask[:, 0] != contact_mask[:, 2])

    return (trot_reward.float() + gallop_reward.float()) * is_command_active


def foot_edge_contact(
    env: ManagerBasedRLEnv,
    # contact sensor -> contact mask
    sensor_cfg: SceneEntityCfg, 
    contact_threshold: float,
    # feet raycast sensor -> feet edge detecter
    FL_foot_edge_detecter: SceneEntityCfg,
    FR_foot_edge_detecter: SceneEntityCfg,
    RL_foot_edge_detecter: SceneEntityCfg,
    RR_foot_edge_detecter: SceneEntityCfg,
    # target minimum height
    height_threshold: float,
    cnt_threshold: int,
) -> torch.Tensor:
    # Get contact sensor states
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_mask = (net_contact_forces[:, 0, sensor_cfg.body_ids, 2] > contact_threshold)

    # foot scanner height(world-frame)
    FL_foot_sensor: RayCaster = env.scene.sensors[FL_foot_edge_detecter.name]
    FR_foot_sensor: RayCaster = env.scene.sensors[FR_foot_edge_detecter.name]
    RL_foot_sensor: RayCaster = env.scene.sensors[RL_foot_edge_detecter.name]
    RR_foot_sensor: RayCaster = env.scene.sensors[RR_foot_edge_detecter.name]
    FL_all = FL_foot_sensor.data.ray_hits_w[..., 2]
    FR_all = FR_foot_sensor.data.ray_hits_w[..., 2]
    RL_all = RL_foot_sensor.data.ray_hits_w[..., 2]
    RR_all = RR_foot_sensor.data.ray_hits_w[..., 2]
    FL_mean = torch.mean(FL_all, dim=1).unsqueeze(1)
    FR_mean = torch.mean(FR_all, dim=1).unsqueeze(1)
    RL_mean = torch.mean(RL_all, dim=1).unsqueeze(1)
    RR_mean = torch.mean(RR_all, dim=1).unsqueeze(1)

    # foot edge
    foot_edge_list = [
        torch.sum(torch.abs(FL_all - FL_mean) > height_threshold, dim=1),
        torch.sum(torch.abs(FR_all - FR_mean) > height_threshold, dim=1),
        torch.sum(torch.abs(RL_all - RL_mean) > height_threshold, dim=1),
        torch.sum(torch.abs(RR_all - RR_mean) > height_threshold, dim=1)
    ]
    foot_edge = torch.stack(foot_edge_list, dim=1)  # 形状: (num_envs, 4)
    foot_edge = (foot_edge >= cnt_threshold).float()

    # penalty
    penalty = torch.sum(foot_edge*contact_mask, dim=1)
    return penalty


def undesired_contacts_curriculum(
    env: ManagerBasedRLEnv, 
    threshold: float, 
    sensor_cfg: SceneEntityCfg,
    base_weight: float = 0.5,
    curriculum_threshold: float = 0.7,
) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold

    # get curriculum progress
    current_curriculum_progress = torch.mean(env.scene.terrain.terrain_levels.float())
    curriculum_max_value = env.scene.terrain.max_terrain_level
    curriculum_ratio = (current_curriculum_progress / curriculum_max_value).clamp(0.0, 1.0)

    dynamic_weight = (curriculum_ratio - curriculum_threshold).clamp_min(0.0) * (1.0/(1.0 - curriculum_threshold))
    total_weight = base_weight + dynamic_weight

    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1) * total_weight

