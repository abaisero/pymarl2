import copy
from types import SimpleNamespace
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from components.episode_buffer import EpisodeBatch
from controllers.basic_controller import BasicMAC
from modules.mixers.qfix import QFix
from modules.mixers.qfix_sum_alt import QFixSumAlt
from utils.logging import Logger
from utils.rl_utils import build_td_lambda_targets
from utils.th_utils import get_parameters_num


def make_mixer(args: SimpleNamespace) -> nn.Module:
    if args.mixer == "qfix":
        return QFix(args)

    if args.mixer == "qfix_sum_alt":
        return QFixSumAlt(args)

    raise ValueError(f'invalid mixer type "{args.mixer}"')


def get_individual_qvalues(mac: BasicMAC, batch: EpisodeBatch) -> torch.Tensor:
    mac.init_hidden(batch.batch_size)
    return torch.stack(
        [mac.forward(batch, t=t) for t in range(batch.max_seq_length)],
        dim=1,
    )


class QFixLearner:
    def __init__(
        self,
        mac: BasicMAC,
        _: dict,  # scheme
        logger: Logger,
        args: SimpleNamespace,
    ):
        self.args = args
        self.logger = logger

        self.mac = mac
        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.mixer = make_mixer(args)
        self.target_mixer = copy.deepcopy(self.mixer)

        self.params = list(mac.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(params=self.params, lr=args.lr)

        self.num_timestep_log_stats = 0
        self.num_episode_target_update = args.target_update_interval

        print("Mixer Size: ")
        print(get_parameters_num(self.mixer.parameters()))

    def train(
        self,
        batch: EpisodeBatch,
        num_timestep: int,
        num_episode: int,
    ):
        # Get the relevant quantities
        states = cast(torch.Tensor, batch["state"])
        # states.shape = (B, T, S)
        actions = cast(torch.Tensor, batch["actions"])
        # actions.shape = (B, T, N, 1)
        rewards = cast(torch.Tensor, batch["reward"])
        # rewards.shape = (B, T, 1)
        terminated = cast(torch.Tensor, batch["terminated"][:, :-1]).float()
        # terminated.shape = (B, T-1, 1)
        mask = cast(torch.Tensor, batch["filled"][:, :-1]).float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        # mask.shape = (B, T-1, 1)
        available_actions = batch["avail_actions"]
        # available_actions.shape = (B, T, N, A)
        actions_onehot = batch["actions_onehot"]
        # actions_onehot.shape = (B, T, N, A)

        unavailable_actions_mask = available_actions == 0
        # unavailable_actions_mask.shape = (B, T, N, A)

        # Calculate estimated Q-Values
        individual_qvalues = get_individual_qvalues(self.mac, batch)
        # individual_qvalues.shape = (B, T, N, A)
        individual_qvalues[unavailable_actions_mask] = -9999999

        B, T, N, A = individual_qvalues.shape

        # Pick the Q-Values for the actions taken by each agent
        chosen_individual_qvalues = torch.gather(
            individual_qvalues,
            dim=-1,
            index=actions,
        ).squeeze(-1)
        # chosen_individual_qvalues.shape = (B, T, N)

        individual_vvalues, maximal_actions = individual_qvalues.max(dim=-1)
        # individual_vvalues.shape = (B, T, N)
        # maximal_actions.shape = (B, T, N)
        maximal_actions = maximal_actions.unsqueeze(-1)
        # maximal_actions.shape = (B, T, N, 1)

        maximal_actions_onehot = F.one_hot(
            maximal_actions,
            self.args.n_actions,
        ).to(torch.device("cuda"), torch.float)
        # maximal_actions_onehot.shape = (B, T, N, A)

        is_action_maximal = actions[:, :-1] == maximal_actions[:, :-1]
        # is_action_maximal.shape = (B, T-1, N, 1)

        # Calculate the Q-Values necessary for the target
        target_individual_qvalues = get_individual_qvalues(self.target_mac, batch)
        # target_individual_qvalues.shape = (B, T, N, A)
        target_individual_qvalues[unavailable_actions_mask] = -9999999

        target_individual_vvalues = target_individual_qvalues.max(dim=-1).values
        # target_individual_vvalues.shape = (B, T, N)

        # Max over target Q-Values
        # Get actions that maximise live Q (for double q-learning)
        # This is the target model evaluated using the non-target maximal actions, per double-Q
        target_maximal_individual_qvalues = torch.gather(
            target_individual_qvalues,
            dim=-1,
            index=maximal_actions,
        ).squeeze(-1)
        # target_maximal_individual_qvalues.shape = (B, T, N)

        chosen_joint_qvalues = self.mixer(
            chosen_individual_qvalues[:, :-1],
            states[:, :-1],
            actions_onehot[:, :-1],
            individual_vvalues[:, :-1],
        )
        # chosen_joint_qvalues.shape = (B, T-1, 1)

        target_maximal_joint_qvalues = self.target_mixer(
            target_maximal_individual_qvalues,
            states,
            maximal_actions_onehot,
            target_individual_vvalues,
        )
        # target_maximal_joint_qvalues.shape = (B, T, 1)

        # Calculate 1-step Q-Learning targets
        target_joint_qvalues = build_td_lambda_targets(
            rewards[:, :-1],
            terminated,
            mask,
            target_maximal_joint_qvalues,
            self.args.n_agents,
            self.args.gamma,
            self.args.td_lambda,
        )
        # target_joint_qvalues.shape = (B, T-1, 1)

        # Td-error
        td_error = chosen_joint_qvalues - target_joint_qvalues.detach()

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = 0.5 * (masked_td_error**2).sum() / mask.sum()

        # Q: what does hit_prob mean, just the proportion of max actions?
        # but shouldn't it be the epsilon fraction..?
        masked_hit_prob = torch.mean(is_action_maximal.float(), dim=2) * mask
        hit_prob = masked_hit_prob.sum() / mask.sum()

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimizer.step()

        if num_timestep >= self.num_timestep_log_stats:
            mask_elems = mask.sum().item()
            td_error_abs = masked_td_error.abs().sum().item() / mask_elems
            q_taken_mean = (chosen_joint_qvalues * mask).sum().item() / (
                mask_elems * self.args.n_agents
            )
            target_mean = (target_joint_qvalues * mask).sum().item() / (
                mask_elems * self.args.n_agents
            )

            self.log_stats(
                {
                    "loss": loss.item(),
                    "hit_prob": hit_prob.item(),
                    "grad_norm": grad_norm.item(),
                    "td_error_abs": td_error_abs,
                    "q_taken_mean": q_taken_mean,
                    "target_mean": target_mean,
                },
                num_timestep,
            )

            self.num_timestep_log_stats = num_timestep + self.args.learner_log_interval

        if num_episode >= self.num_episode_target_update:
            self._update_targets()
            self.num_episode_target_update = (
                num_episode + self.args.target_update_interval
            )

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.mixer.cuda()
        self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        torch.save(self.mixer.state_dict(), f"{path}/mixer.th")
        torch.save(self.optimizer.state_dict(), f"{path}/opt.th")

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        self.mixer.load_state_dict(torch.load(f"{path}/mixer.th", "cpu"))
        self.target_mixer.load_state_dict(torch.load(f"{path}/mixer.th", "cpu"))
        self.optimizer.load_state_dict(torch.load(f"{path}/opt.th", "cpu"))

    def log_stats(self, stats: dict, num_timestep):
        for key, value in stats.items():
            self.logger.log_stat(key, value, num_timestep)
