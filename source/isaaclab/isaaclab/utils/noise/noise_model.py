# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.utils.buffers import CircularBuffer

if TYPE_CHECKING:
    from . import noise_cfg

##
# Noise as functions.
##


def constant_noise(data: torch.Tensor, cfg: noise_cfg.ConstantNoiseCfg) -> torch.Tensor:
    """Applies a constant noise bias to a given data set.

    Args:
        data: The unmodified data set to apply noise to.
        cfg: The configuration parameters for constant noise.

    Returns:
        The data modified by the noise parameters provided.
    """

    # fix tensor device for bias on first call and update config parameters
    if isinstance(cfg.bias, torch.Tensor):
        cfg.bias = cfg.bias.to(device=data.device)

    if cfg.operation == "add":
        return data + cfg.bias
    elif cfg.operation == "scale":
        return data * cfg.bias
    elif cfg.operation == "abs":
        return torch.zeros_like(data) + cfg.bias
    else:
        raise ValueError(f"Unknown operation in noise: {cfg.operation}")


def uniform_noise(data: torch.Tensor, cfg: noise_cfg.UniformNoiseCfg) -> torch.Tensor:
    """Applies a uniform noise to a given data set.

    Args:
        data: The unmodified data set to apply noise to.
        cfg: The configuration parameters for uniform noise.

    Returns:
        The data modified by the noise parameters provided.
    """

    # fix tensor device for n_max on first call and update config parameters
    if isinstance(cfg.n_max, torch.Tensor):
        cfg.n_max = cfg.n_max.to(data.device)
    # fix tensor device for n_min on first call and update config parameters
    if isinstance(cfg.n_min, torch.Tensor):
        cfg.n_min = cfg.n_min.to(data.device)

    if cfg.operation == "add":
        return data + torch.rand_like(data) * (cfg.n_max - cfg.n_min) + cfg.n_min
    elif cfg.operation == "scale":
        return data * (torch.rand_like(data) * (cfg.n_max - cfg.n_min) + cfg.n_min)
    elif cfg.operation == "abs":
        return torch.rand_like(data) * (cfg.n_max - cfg.n_min) + cfg.n_min
    else:
        raise ValueError(f"Unknown operation in noise: {cfg.operation}")


def gaussian_noise(data: torch.Tensor, cfg: noise_cfg.GaussianNoiseCfg) -> torch.Tensor:
    """Applies a gaussian noise to a given data set.

    Args:
        data: The unmodified data set to apply noise to.
        cfg: The configuration parameters for gaussian noise.

    Returns:
        The data modified by the noise parameters provided.
    """

    # fix tensor device for mean on first call and update config parameters
    if isinstance(cfg.mean, torch.Tensor):
        cfg.mean = cfg.mean.to(data.device)
    # fix tensor device for std on first call and update config parameters
    if isinstance(cfg.std, torch.Tensor):
        cfg.std = cfg.std.to(data.device)

    if cfg.operation == "add":
        return data + cfg.mean + cfg.std * torch.randn_like(data)
    elif cfg.operation == "scale":
        return data * (cfg.mean + cfg.std * torch.randn_like(data))
    elif cfg.operation == "abs":
        return cfg.mean + cfg.std * torch.randn_like(data)
    else:
        raise ValueError(f"Unknown operation in noise: {cfg.operation}")


##
# Noise models as classes
##


class NoiseModel:
    """Base class for noise models."""

    def __init__(self, noise_model_cfg: noise_cfg.NoiseModelCfg, num_envs: int, device: str):
        """Initialize the noise model.

        Args:
            noise_model_cfg: The noise configuration to use.
            num_envs: The number of environments.
            device: The device to use for the noise model.
        """
        self._noise_model_cfg = noise_model_cfg
        self._num_envs = num_envs
        self._device = device

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset the noise model.

        This method can be implemented by derived classes to reset the noise model.
        This is useful when implementing temporal noise models such as random walk.

        Args:
            env_ids: The environment ids to reset the noise model for. Defaults to None,
                in which case all environments are considered.
        """
        pass

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the noise to the data.

        Args:
            data: The data to apply the noise to. Shape is (num_envs, ...).

        Returns:
            The data with the noise applied. Shape is the same as the input data.
        """
        return self._noise_model_cfg.noise_cfg.func(data, self._noise_model_cfg.noise_cfg)


class NoiseModelWithAdditiveBias(NoiseModel):
    """Noise model with an additive bias.

    The bias term is sampled from a the specified distribution on reset.
    """

    def __init__(self, noise_model_cfg: noise_cfg.NoiseModelWithAdditiveBiasCfg, num_envs: int, device: str):
        # initialize parent class
        super().__init__(noise_model_cfg, num_envs, device)
        # store the bias noise configuration
        self._bias_noise_cfg = noise_model_cfg.bias_noise_cfg
        self._bias = torch.zeros((num_envs, 1), device=self._device)
        self._num_components: int | None = None
        self._sample_bias_per_component = noise_model_cfg.sample_bias_per_component

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset the noise model.

        This method resets the bias term for the specified environments.

        Args:
            env_ids: The environment ids to reset the noise model for. Defaults to None,
                in which case all environments are considered.
        """
        # resolve the environment ids
        if env_ids is None:
            env_ids = slice(None)
        # reset the bias term
        self._bias[env_ids] = self._bias_noise_cfg.func(self._bias[env_ids], self._bias_noise_cfg)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply bias noise to the data.

        Args:
            data: The data to apply the noise to. Shape is (num_envs, ...).

        Returns:
            The data with the noise applied. Shape is the same as the input data.
        """
        # if sample_bias_per_component, on first apply, expand bias to match last dim of data
        if self._sample_bias_per_component and self._num_components is None:
            *_, self._num_components = data.shape
            # expand bias from (num_envs,1) to (num_envs, num_components)
            self._bias = self._bias.repeat(1, self._num_components)
            # now re-sample that expanded bias in-place
            self.reset()
        return super().__call__(data) + self._bias


class NoiseModelWithPeriodicBias(NoiseModel):
    """Noise model with an additive bias that is periodically resampled.

    Unlike :class:`NoiseModelWithAdditiveBias` which resamples bias on every reset,
    this class resamples the bias at random intervals within a specified time range.
    The bias for each environment is resampled independently after a randomly chosen
    time interval uniformly sampled from ``bias_resample_interval``.

    This is useful for simulating slowly varying bias disturbances that change
    periodically rather than at every environment reset.

    Note:
        The ``__call__`` method automatically updates the time tracking and resamples
        the bias when needed based on the configured ``dt`` parameter.
    """

    def __init__(self, noise_model_cfg: noise_cfg.NoiseModelWithPeriodicBiasCfg, num_envs: int, device: str):
        # initialize parent class
        super().__init__(noise_model_cfg, num_envs, device)
        # store the bias noise configuration
        self._bias_noise_cfg = noise_model_cfg.bias_noise_cfg
        self._bias = torch.zeros((num_envs, 1), device=self._device)
        self._num_components: int | None = None
        self._sample_bias_per_component = noise_model_cfg.sample_bias_per_component

        # periodic resampling parameters
        self._bias_resample_min_interval, self._bias_resample_max_interval = noise_model_cfg.bias_resample_interval
        self._dt = noise_model_cfg.dt

        # track time since last bias resampling for each environment
        self._time_since_last_resample = torch.zeros(num_envs, device=self._device)
        # track next resample time for each environment
        self._next_resample_time = torch.zeros(num_envs, device=self._device)

        # sample random intervals for each environment
        random_intervals = torch.rand(self._num_envs, device=self._device)
        random_intervals = random_intervals * (self._bias_resample_max_interval - self._bias_resample_min_interval)
        random_intervals = random_intervals + self._bias_resample_min_interval
        self._next_resample_time = random_intervals

    def _update_and_resample_bias(self):
        """Update time tracking and resample bias if needed."""
        # update time since last resample
        self._time_since_last_resample += self._dt

        # check which environments need resampling
        should_resample = self._time_since_last_resample >= self._next_resample_time

        if torch.any(should_resample):
            # get environment ids that need resampling
            env_ids = torch.where(should_resample)[0]

            # resample bias
            self._bias[env_ids] = self._bias_noise_cfg.func(self._bias[env_ids], self._bias_noise_cfg)

            # reset time since last resample
            self._time_since_last_resample[env_ids] = 0.0

            # sample new random resample intervals
            random_intervals = torch.rand(len(env_ids), device=self._device)
            random_intervals = random_intervals * (self._bias_resample_max_interval - self._bias_resample_min_interval)
            random_intervals = random_intervals + self._bias_resample_min_interval
            self._next_resample_time[env_ids] = random_intervals

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset the noise model.

        This method resets the bias term and timing for the specified environments.

        Args:
            env_ids: The environment ids to reset the noise model for. Defaults to None,
                in which case all environments are considered.
        """
        # resolve the environment ids
        if env_ids is None:
            env_ids = slice(None)
        # reset the bias term
        self._bias[env_ids] = self._bias_noise_cfg.func(self._bias[env_ids], self._bias_noise_cfg)
        # reset timing counters
        self._time_since_last_resample[env_ids] = 0.0
        # sample new random resample intervals for the specified environments
        # Handle both slice and array indexing
        if isinstance(env_ids, slice):
            num_resets = self._num_envs
        else:
            num_resets = len(env_ids)
        random_intervals = torch.rand(num_resets, device=self._device)
        random_intervals = random_intervals * (self._bias_resample_max_interval - self._bias_resample_min_interval)
        random_intervals = random_intervals + self._bias_resample_min_interval
        self._next_resample_time[env_ids] = random_intervals

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply bias noise to the data.

        This method automatically updates the time tracking and resamples the bias
        when needed based on the configured ``dt`` parameter.

        Args:
            data: The data to apply the noise to. Shape is (num_envs, ...).

        Returns:
            The data with the noise applied. Shape is the same as the input data.
        """
        # if sample_bias_per_component, on first apply, expand bias to match last dim of data
        if self._sample_bias_per_component and self._num_components is None:
            *_, self._num_components = data.shape
            # expand bias from (num_envs,1) to (num_envs, num_components)
            self._bias = self._bias.repeat(1, self._num_components)
            # now re-sample that expanded bias in-place
            self.reset()

        # update time tracking and resample bias if needed
        self._update_and_resample_bias()

        # apply bias noise to the data
        return super().__call__(data) + self._bias
