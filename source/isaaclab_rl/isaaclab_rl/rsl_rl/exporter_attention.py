# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os
import torch


def export_policy_as_jit_attention(
        policy: object, 
        normalizer_basic: object | None, 
        normalizer_scan: object | None, 
        path: str, filename="policy.pt"
    ):
    """Export policy into a Torch JIT file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".
    """
    policy_exporter = _TorchPolicyExporterAttention(policy, normalizer_basic, normalizer_scan)
    policy_exporter.export(path, filename)


"""
Helper Classes - Private.
"""


class _TorchPolicyExporterAttention(torch.nn.Module):
    """Exporter of actor-critic into JIT file."""

    def __init__(self, policy, normalizer_basic=None, normalizer_scan=None):
        super().__init__()
        self.is_recurrent = policy.is_recurrent
        # copy policy parameters
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
        else:
            raise ValueError("Policy does not have an actor/student module.")
        # copy normalizer if exists
        if normalizer_basic:
            self.normalizer_basic = copy.deepcopy(normalizer_basic)
        else:
            self.normalizer_basic = torch.nn.Identity()
        if normalizer_scan:
            self.normalizer_scan = copy.deepcopy(normalizer_scan)
        else:
            self.normalizer_scan = torch.nn.Identity()

    def forward(self, basic_obs, scan_obs):
        return self.actor(self.normalizer_basic(basic_obs), self.normalizer_scan(scan_obs))

    @torch.jit.export
    def reset(self):
        pass

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

