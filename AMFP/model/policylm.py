import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
import gym
import d4rl
from tqdm import tqdm
from torch import Tensor
from .utils import get_return, suppress_output
import numpy as np


class PolicyLM(pl.LightningModule):
    def __init__(self, seed: int, env_name: str, obs_dim: int, action_dim: int, return_dim: int, lr: float, use_scheduler: bool, epochs: int, ctx_size: int,
                 future_horizon: int, num_eval_rollouts: int, eval_last_k: int, return_type: str, model_config: DictConfig):
        super().__init__()

        self.seed = seed
        self.env_name = env_name
        self.ctx_size = ctx_size
        self.future_horizon = future_horizon
        self.lr = lr
        self.use_scheduler = use_scheduler
        self.num_epochs = epochs
        self.eval_last_k = eval_last_k
        self.return_type = return_type
        self.num_eval_rollouts = num_eval_rollouts

        if self.return_type == 'rtg':  # return as expected return
            self.return_fracs = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
            if 'halfcheetah-medium' in self.env_name:
                self.return_fracs += [0.4, 0.5]
        else:  # return as target state
            self.return_fracs = [1]  # a placeholder

        self.env = gym.make(self.env_name)
        with suppress_output():
            self.env.seed(seed=self.seed)
            self.env.action_space.seed(seed=self.seed)
            self.env.observation_space.seed(seed=self.seed)
        self.eval_score = {frac: [] for frac in self.return_fracs}

        self.model = None

    def on_validation_epoch_end(self):
        if self.current_epoch + 1 > self.num_epochs - self.eval_last_k:  # start evaluations for the last k checkpoints
            eval_horizon = self.env._max_episode_steps

            for frac in tqdm(self.return_fracs, leave=False, desc=f'Validation Rollout {self.current_epoch}'):
                rollout_scores = torch.zeros(size=(self.num_eval_rollouts,), device=self.device)
                for i in tqdm(range(self.num_eval_rollouts), leave=False, desc=f'Retrun Frac {frac}'):
                    with suppress_output():
                        ini_obs = self.env.reset()
                        _return = get_return(env=self.env, return_type=self.return_type, return_frac=frac, obs=ini_obs)

                    ini_obs = torch.tensor(ini_obs, device=self.device, dtype=torch.float)[:self.model.obs_dim]
                    _return = torch.tensor(_return, device=self.device, dtype=torch.float)
                    obs, actions, _return = self.init_eval(ini_obs, _return, self.model.obs_dim, self.model.action_dim, self.model.rtg_dim)
                    for t in range(eval_horizon):
                        a = self.ar_step(t, obs, actions, _return)
                        next_obs, reward, terminal, _ = self.env.step(a.cpu().numpy())
                        rollout_scores[i] += reward

                        if terminal:
                            break

                        next_obs = torch.tensor(next_obs, device=self.device, dtype=torch.float)[:self.model.obs_dim]
                        obs, actions = self.ar_step_end(t, next_obs, a, obs, actions)

                    rollout_scores[i] = self.env.get_normalized_score(rollout_scores[i])
                self.eval_score[frac].append(rollout_scores.mean().item())

            self.log_dict(
                {f'eval/norm_score_{frac}' if self.return_type == 'rtg' else 'eval/norm_score': self.eval_score[frac][-1] for frac in self.eval_score.keys()},sync_dist=True)

        if self.current_epoch + 1 == self.num_epochs:  # get the best evaluation result among the last k checkpoints
            self.log_dict(
                {f'result/norm_score_{frac}' if self.return_type == 'rtg' else 'result/norm_score': np.max(self.eval_score[frac][-self.eval_last_k:]) for frac in
                 self.eval_score.keys()},sync_dist=True)

    # eval funcs
    def init_eval(self, ini_obs: Tensor, _return: Tensor, obs_dim: int, action_dim: int, rtg_dim: int):
        raise NotImplementedError

    def ar_step(self, timestep: int, observations: Tensor, actions: Tensor, _return: Tensor):
        raise NotImplementedError

    def ar_step_end(self, timestep: int, next_obs: Tensor, action: Tensor, obs_seq: Tensor, action_seq: Tensor):
        raise NotImplementedError