# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg

from isaaclab_tasks.manager_based.locomotion.velocity.mdp.symmetry import go2_skill_walk


@configclass
class UnitreeGo2LocoSkillPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    # max_iterations = 4001   # teacher - stage1
    max_iterations = 6001   # teacher - stage2
    save_interval = 250
    experiment_name = "Go2-Loco-Skill-Walk"
    obs_groups = {
        "policy": ["proprioception", "mapScans", "privileged"],
        "critic": ["proprioception", "mapScans", "privileged"],
    }
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        noise_std_type="scalar",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        rnd_cfg=None,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True, 
            data_augmentation_func=go2_skill_walk.compute_symmetric_states,
            use_mirror_loss=True,
            mirror_loss_coeff=0.1,
        ),
    )
