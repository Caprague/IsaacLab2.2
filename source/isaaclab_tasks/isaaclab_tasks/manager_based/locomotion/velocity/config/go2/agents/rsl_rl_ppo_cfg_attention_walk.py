# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfgAttention, RslRlPpoActorCriticCfgAttention, RslRlPpoAlgorithmCfgAttention, RslRlSymmetryCfg

from isaaclab_tasks.manager_based.locomotion.velocity.mdp.symmetry import go2_attention_walk


@configclass
class UnitreeGo2LocoSkillPPORunnerCfg(RslRlOnPolicyRunnerCfgAttention):
    num_steps_per_env = 24
    # max_iterations = 7001   # stage1 enable
    max_iterations = 14001   # stage2 enable
    save_interval = 250
    experiment_name = "Go2-Loco-Attention-Walk"
    obs_groups = {
        "basic": ["proprioception"],
        "scan": ["mapScans"],
        "critic": ["proprioception", "privileged", "mapScansCritic"],
    }
    policy = RslRlPpoActorCriticCfgAttention(
        init_noise_std=1.0,
        basic_obs_normalization=False,
        scan_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        # Scan 尺寸(长x宽点数)
        scan_size_x=17,
        scan_size_y=11,
        # 用于实现MHA模块，需要设定的参数
        mha_embed_dim=64,
        mha_num_heads=8,
        mha_dropout=0.05,
        # 本体感知的各观测项维度，及历史观测长度
        basic_obs_dims_list=[3, 3, 3, 12, 12, 12],
        basic_obs_history_length=3,
    )
    algorithm = RslRlPpoAlgorithmCfgAttention(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        rnd_cfg=None,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True, 
            data_augmentation_func=go2_attention_walk.compute_symmetric_states,
            # use_mirror_loss=True,
            # mirror_loss_coeff=0.1,
        ),
    )
