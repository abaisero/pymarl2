from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from .qfix_weight import QFix_FF_Weight, QFix_SI_Weight, gt_constraint
from .qmix import QMixer
from .vdn import VDNMixer


def make_inner_mixer(inner_args: SimpleNamespace, args: SimpleNamespace) -> nn.Module:
    """Create the inner mixer module, which satisfies IGM but is not IGM-complete."""
    if inner_args.mixer == "vdn":
        return VDNMixer()

    if inner_args.mixer == "qmix":
        inner_args.n_agents = args.n_agents
        inner_args.n_actions = args.n_actions
        inner_args.state_shape = args.state_shape
        return QMixer(inner_args)

    raise ValueError(f'invalid inner mixer type "{inner_args.mixer}"')


class QFix(nn.Module):
    def __init__(self, args: SimpleNamespace):
        super().__init__()

        self.args = args

        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = np.prod(args.state_shape).item()
        self.joint_action_dim = args.n_agents * args.n_actions

        inner_args = SimpleNamespace(**args.inner_mixer)
        self.inner_mixer = make_inner_mixer(inner_args, args)

        # NOTE: w is unconstrained here, the constraint is applied later
        self.w_module = (
            QFix_SI_Weight(args, multi_output=False)
            if args.qfix_w_attention
            else QFix_FF_Weight(args, multi_output=False)
        )
        self.b_module = nn.Sequential(
            nn.Linear(self.state_dim, args.hypernet_embed),
            nn.ReLU(),
            nn.Linear(args.hypernet_embed, 1),
        )

    def forward(
        self,
        individual_qvalues: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        individual_vvalues: torch.Tensor,
    ) -> torch.Tensor:
        """
        Applies QFIX mixing.

        :param individual_qvalues: FloatTensor, shape = (B, T, N)
        :param states: FloatTensor, shape = (B, T, S)
        :param actions: FloatTensor, shape = (B, T, N, A), onehot encoding
        :param individual_vvalues: FloatTensor, shape = (B, T, N)
        :return: FloatTensor, shape = (B, T, 1)
        """

        # store batch size
        batch_size = individual_qvalues.size(0)

        inner_qvalues: torch.Tensor = self.inner_mixer(individual_qvalues, states)
        inner_vvalues: torch.Tensor = self.inner_mixer(individual_vvalues, states)
        inner_qvalues = inner_qvalues.view(-1, 1)
        inner_vvalues = inner_vvalues.view(-1, 1)

        # flatten batch and time dimensions
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.joint_action_dim)
        w = gt_constraint(
            self.w_module(states, actions) + self.args.qfix_w_delta,
            self.args.qfix_w_gt,
        )
        b = self.b_module(states)
        outputs = self._forward(inner_qvalues, inner_vvalues, w, b)

        # restore batch dimension
        return outputs.view(batch_size, -1, 1)

    def _forward(
        self,
        inner_qvalues: torch.Tensor,
        inner_vvalues: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        if self.args.qfix_type == "qfix":
            return self._forward_qfix(inner_qvalues, inner_vvalues, w, b)

        if self.args.qfix_type == "q+fix":
            return self._forward_additive_qfix(inner_qvalues, inner_vvalues, w, b)

        raise ValueError(f"Invalid {self.args.qfix_type=}")

    def _forward_qfix(
        self,
        inner_qvalues: torch.Tensor,
        inner_vvalues: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        inner_advantages = inner_qvalues - inner_vvalues
        return w * inner_advantages + b

    def _forward_additive_qfix(
        self,
        inner_qvalues: torch.Tensor,
        inner_vvalues: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        inner_advantages = inner_qvalues - inner_vvalues

        if self.args.qfix_detach_advantages:
            inner_advantages = inner_advantages.detach()

        return inner_qvalues + w * inner_advantages + b
