# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Functions to specify the symmetry in the observation and action space for ANYmal."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

# specify the functions that are available for import
__all__ = ["compute_symmetric_states"]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augments the given observations and actions by applying symmetry transformations.

    This function creates augmented versions of the provided observations and actions by applying
    four symmetrical transformations: original, left-right, front-back, and diagonal. The symmetry
    transformations are beneficial for reinforcement learning tasks by providing additional
    diverse data without requiring additional data collection.

    Args:
        env: The environment instance.
        obs: The original observation tensor dictionary. Defaults to None.
        actions: The original actions tensor. Defaults to None.

    Returns:
        Augmented observations and actions tensors, or None if the respective input was None.
    """

    # observations
    if obs is not None:
        batch_size = obs.batch_size[0]
        # go2 have 2 different symmetries
        obs_aug = obs.repeat(2)

        # proprioception observation group
        # -- original
        obs_aug["proprioception"][:batch_size] = obs["proprioception"][:]
        # -- left-right
        obs_aug["proprioception"][batch_size:] = _transform_proprioception_obs_left_right(env.unwrapped, obs["proprioception"])
        # mapScans observation group
        # -- original
        obs_aug["mapScans"][:batch_size] = obs["mapScans"][:]
        # -- left-right
        obs_aug["mapScans"][batch_size:] = _transform_mapScans_3d_obs_left_right(env.unwrapped, obs["mapScans"])
        # privileged observation group
        # -- original
        obs_aug["privileged"][:batch_size] = obs["privileged"][:]
        # -- left-right
        obs_aug["privileged"][batch_size:] = _transform_privileged_obs_left_right(env.unwrapped, obs["privileged"])

    else:
        obs_aug = None

    # actions
    if actions is not None:
        batch_size = actions.shape[0]
        # go2 have 2 different symmetries
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        # -- original
        actions_aug[:batch_size] = actions[:]
        # -- left-right
        actions_aug[batch_size:] = _transform_actions_left_right(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


"""
Symmetry functions for observations.
"""


def _transform_proprioception_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    """Apply a left-right symmetry transformation to the observation tensor.

    This function modifies the given observation tensor by applying transformations
    that represent a symmetry with respect to the left-right axis. This includes
    negating certain components of the linear and angular velocities, projected gravity,
    velocity commands, and flipping the joint positions, joint velocities, and last actions
    for the ANYmal robot. Additionally, if height-scan data is present, it is flipped
    along the relevant dimension.

    Args:
        env: The environment instance from which the observation is obtained.
        obs: The observation tensor to be transformed.

    Returns:
        The transformed observation tensor with left-right symmetry applied.
    """
    # copy observation tensor
    obs = obs.clone()
    device = obs.device
    # velocity commands 9
    obs[:, :9] = obs[:, :9] * torch.tensor([1, -1, -1, 1, -1, -1, 1, -1, -1], device=device)
    # imu ang vel 9
    obs[:, 9:18] = obs[:, 9:18] * torch.tensor([-1, 1, -1, -1, 1, -1, -1, 1, -1], device=device)
    # projected gravity 9
    obs[:, 18:27] = obs[:, 18:27] * torch.tensor([1, -1, 1, 1, -1, 1, 1, -1, 1], device=device)
    # joint pos rel 36
    obs[:, 27:63] = _switch_go2_joints_left_right_vectorized(obs[:, 27:63], repeat=3)
    # joint vel rel 36
    obs[:, 63:99] = _switch_go2_joints_left_right_vectorized(obs[:, 63:99], repeat=3)
    # actions 36
    obs[:, 99:135] = _switch_go2_joints_left_right_vectorized(obs[:, 99:135], repeat=3)

    return obs


def _transform_mapScans_3d_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    # note: this is hard-coded for grid-pattern of ordering "xy" and size (1.6, 1.0)
    original_shape = obs.shape
    obs = obs.view(-1, 11, 17, 3).flip(dims=[1]).reshape(original_shape)
    return obs


def _transform_privileged_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    # copy observation tensor
    obs = obs.clone()
    device = obs.device
    # base lin vel 9
    obs[:, :9] = obs[:, :9] * torch.tensor([1, -1, 1, 1, -1, 1, 1, -1, 1], device=device)
    # feet distance 12
    obs[:, 9:21] = _switch_go2_feets_left_right_vectorized(obs[:, 9:21], repeat=3)
    # feet contact mask 12
    obs[:, 21:33] = _switch_go2_feets_left_right_vectorized(obs[:, 21:33], repeat=3)
    # feet air height 12
    obs[:, 33:45] = _switch_go2_feets_left_right_vectorized(obs[:, 33:45], repeat=3)
    
    return obs


"""
Symmetry functions for actions.
"""


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the actions tensor.

    This function modifies the given actions tensor by applying transformations
    that represent a symmetry with respect to the left-right axis. This includes
    flipping the joint positions, joint velocities, and last actions for the
    ANYmal robot.

    Args:
        actions: The actions tensor to be transformed.

    Returns:
        The transformed actions tensor with left-right symmetry applied.
    """
    actions = actions.clone()
    actions[:] = _switch_go2_joints_left_right_vectorized(actions[:], repeat=1)
    return actions


"""
Helper functions for symmetry.

In Isaac Sim, the joint ordering is as follows:
[
    'FL_hip', 'FR_hip', 'RL_hip', 'RR_hip',
    'FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh',
    'FL_calf', 'FR_calf', 'RL_calf', 'RR_calf'
]

Correspondingly, the joint ordering for the Go2 robot is:

* FL = left front --> [0, 4, 8]
* FR = right front --> [1, 5, 9]
* RL = left hind --> [2, 6, 10]
* RR = right hind --> [3, 7, 11]
"""

def _switch_go2_joints_left_right_vectorized(joint_data: torch.Tensor, repeat: int) -> torch.Tensor:
    """向量化实现左-右对称变换，避免CPU循环，充分利用GPU并行计算[1,5](@ref)
    
    Args:
        joint_data: 输入张量，形状为(..., 12 * repeat)
        repeat: 关节观测组的数量
    
    Returns:
        应用左-右对称变换后的张量，形状与输入相同
    """
    original_shape = joint_data.shape
    # 重塑为(..., repeat, 12)以便向量化操作[7](@ref)
    joint_data_reshaped = joint_data.view(*original_shape[:-1], repeat, 12)
    
    # 创建左右关节交换的索引映射[3](@ref)
    # 左关节索引: 0,4,8,2,6,10 → 对应右关节: 1,5,9,3,7,11
    # 右关节索引: 1,5,9,3,7,11 → 对应左关节: 0,4,8,2,6,10
    left_indices = torch.tensor([0, 4, 8, 2, 6, 10], device=joint_data.device)
    right_indices = torch.tensor([1, 5, 9, 3, 7, 11], device=joint_data.device)
    
    # 创建交换后的张量 - 预分配内存[8](@ref)
    joint_data_switched = torch.empty_like(joint_data_reshaped)
    
    # 向量化交换操作[1,5](@ref)
    # 左关节 ← 右关节对应值
    joint_data_switched[..., left_indices] = joint_data_reshaped[..., right_indices]
    # 右关节 ← 左关节对应值  
    joint_data_switched[..., right_indices] = joint_data_reshaped[..., left_indices]
    
    # 向量化符号翻转：髋关节(0,1,2,3)乘以-1[8](@ref)
    hip_indices = torch.tensor([0, 1, 2, 3], device=joint_data.device)
    joint_data_switched[..., hip_indices] = joint_data_switched[..., hip_indices] * -1.0
    
    # 重塑回原始形状
    return joint_data_switched.view(original_shape)


def _switch_go2_feets_left_right_vectorized(feet_data: torch.Tensor, repeat: int) -> torch.Tensor:
    original_shape = feet_data.shape
    feet_data_reshaped = feet_data.view(*original_shape[:-1], repeat, 4)
    
    left_indices = torch.tensor([0, 2], device=feet_data.device)
    right_indices = torch.tensor([1, 3], device=feet_data.device)
    
    feet_data_switched = torch.empty_like(feet_data_reshaped)
    
    feet_data_switched[..., left_indices] = feet_data_reshaped[..., right_indices]
    feet_data_switched[..., right_indices] = feet_data_reshaped[..., left_indices]
    
    return feet_data_switched.view(original_shape)

