import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from omegaconf import DictConfig
from .amfpnet import AMFPNet, AggSlotMAE
from .policylm import PolicyLM
from .utils import Device


class MLPBNPolicy(nn.Module):
    def __init__(self, aggslotmae: AggSlotMAE, config: DictConfig):
        super().__init__()
        self.aggslotmae = aggslotmae
        self.embed_dim = config.embed_dim
        self.obs_dim = self.aggslotmae.dim['states']
        self.action_dim = self.aggslotmae.dim['actions']
        self.rtg_dim = self.aggslotmae.dim['returns']
        self.n_slots = self.aggslotmae.n_slots

        for param in self.aggslotmae.parameters():
            param.requires_grad = False
        self.aggslotmae.eval()

        self.obs_return_embed = nn.Linear(self.obs_dim + self.rtg_dim, self.embed_dim)
        self.bn_embed = nn.Linear(self.n_slots * self.aggslotmae.embed_dim, self.embed_dim)
        self.policy = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.ReLU(),
            nn.Dropout(config.pdrop),
            nn.Linear(self.embed_dim, self.action_dim)
        )

    def forward(self, observations: Tensor, _return: Tensor, obs_mask: Tensor):
        batch_size, _, _ = observations.shape
        latent_future = self.aggslotmae.slots_exact(observations, _return, obs_mask)
        o_keep = observations[obs_mask == 0].view(batch_size, -1, self.obs_dim)
        obs_return = torch.cat([o_keep[:, -1:], _return], dim=-1)

        obs_return = F.relu(self.obs_return_embed(obs_return))
        bottleneck = F.relu(self.bn_embed(latent_future.view(batch_size, 1, -1)))
        obs_return_bn = torch.cat([obs_return, bottleneck], dim=-1)
        pred_a = self.policy(obs_return_bn)

        return pred_a


class PolicyNet(PolicyLM):
    def __init__(self, seed: int, env_name: str, obs_dim: int, action_dim: int, return_dim: int, lr: float, use_scheduler: bool, epochs: int, ctx_size: int,
                 future_horizon: int, stage: str, num_eval_rollouts: int, eval_last_k: int, return_type: str, model_config: DictConfig, **kwargs):
        super().__init__(seed, env_name, obs_dim, action_dim, return_dim, lr, use_scheduler, epochs, ctx_size, future_horizon, num_eval_rollouts, eval_last_k,return_type, model_config)

        amfpnet = AMFPNet.load_from_checkpoint(checkpoint_path=model_config.tjn_ckpt_path, map_location=self.device)
        assert self.env_name == amfpnet.env_name and self.ctx_size == amfpnet.ctx_size and self.future_horizon == amfpnet.future_horizon
        aggslotmae = amfpnet.model

        self.stage = stage
        # self.mask_type = amfpnet.mask_type
        self.mask_ratios = model_config.mask_ratios
        self.mask_ratio_weights = model_config.mask_ratio_weights

        self.model = MLPBNPolicy(aggslotmae, model_config)

        self.save_hyperparameters()

    def forward(self, observations: Tensor, _return: Tensor, obs_mask: Tensor):
        return self.model.forward(observations, _return, obs_mask)

    def loss(self, target_a: Tensor, pred_a: Tensor):
        '''
        future_padding_mask: boolean tensor, 0 keep, 1 pad
        '''
        loss = F.mse_loss(pred_a, target_a)  # B x 1 x D

        return loss

    def ar_mask(self, batch_size: int, length: int, keep_len: float, device: Device):
        mask = torch.ones([batch_size, length], device=device)
        mask[:, :keep_len] = 0
        return mask

    def training_step(self, batch, batch_idx):
        observations, actions, _return, _, _ = batch
        batch_size, length, _ = observations.shape
        mask_ratio = np.random.choice(self.mask_ratios, 1, p=self.mask_ratio_weights)[0]

        # For convenience, make sure the number of unmasked (obs/action) is the same across examples when masking
        keep_len = max(1, int(self.ctx_size * (1 - mask_ratio)))
        obs_mask = self.ar_mask(batch_size, length, keep_len, observations.device)

        target_a = actions[:, keep_len - 1].view(batch_size, 1, -1)
        if self.return_type == 'rtg':
            _return = _return[:, keep_len - 1: keep_len]

        pred_a = self(observations, _return, obs_mask)
        loss = self.loss(target_a, pred_a)

        self.log_dict({
            'train/train_loss': loss
        },
            sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        observations, actions, _return, _, _ = batch
        batch_size, length, _ = observations.shape
        mask_ratio = np.random.choice(self.mask_ratios, 1, p=self.mask_ratio_weights)[0]

        # For convenience, make sure the number of unmasked (obs/action) is the same across examples when masking
        keep_len = max(1, int(self.ctx_size * (1 - mask_ratio)))
        obs_mask = self.ar_mask(batch_size, length, keep_len, observations.device)

        target_a = actions[:, keep_len - 1].view(batch_size, 1, -1)
        if self.return_type == 'rtg':
            _return = _return[:, keep_len - 1: keep_len]

        pred_a = self(observations, _return, obs_mask)
        loss = self.loss(target_a, pred_a)

        self.log_dict({
            'val/val_loss': loss
        },
            sync_dist=True)

    def on_validation_epoch_end(self):
        return super().on_validation_epoch_end()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        if self.use_scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=0.5, step_size=self.num_epochs // 2)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}
            }
        return {
            'optimizer': optimizer,
        }

    # eval funcs
    def init_eval(self, ini_obs: Tensor, _return: Tensor, obs_dim: int, action_dim: int, rtg_dim: int):
        # ini_obs: (obs_dim, )
        observations = torch.zeros(size=(1, self.ctx_size, obs_dim), device=self.device)  # 1 x ctx_size x obs_dim
        observations[0, 0] = ini_obs
        # _return: (return_dim, )
        _return = _return.view(1, -1, rtg_dim)
        actions = torch.zeros(size=(1, self.ctx_size, action_dim), device=self.device)  # 1 x ctx_size x action_dim
        return observations, actions, _return

    def make_ar_mask(self, timestep: int):
        '''
        make observation mask for first autoregressive ctx_size steps
        '''
        obs_mask = torch.zeros(size=(1, self.ctx_size), device=self.device)
        action_mask = torch.zeros(size=(1, self.ctx_size), device=self.device)
        action_mask[:, -1] = 1  # the last action is always masked
        if timestep < self.ctx_size - 1:
            obs_mask[0, 1 + timestep:] = 1  # at first ctx_size steps, replace idle observations with obs masked token
            action_mask[0, timestep:] = 1
        return obs_mask, action_mask

    def ar_step(self, timestep: int, observations: Tensor, actions: Tensor, _return: Tensor):
        obs_mask, _ = self.make_ar_mask(timestep)
        pred_a = self(observations, _return, obs_mask)
        action = pred_a[0, 0]
        return action

    def ar_step_end(self, timestep: int, next_obs: Tensor, action: Tensor, obs_seq: Tensor, action_seq: Tensor):
        if timestep < self.ctx_size - 1:
            obs_seq[0, 1 + timestep] = next_obs
            new_obs_seq = obs_seq
            action_seq[0, timestep] = action
            new_action_seq = action_seq
        else:
            next_obs = next_obs.view(1, 1, obs_seq.shape[-1])
            new_obs_seq = torch.cat([obs_seq, next_obs], dim=1)
            action_seq[0, -1] = action
            new_action_seq = torch.cat([action_seq, torch.zeros(size=(1, 1, action_seq.shape[-1]), device=action_seq.device)], dim=1)

        new_obs_seq = new_obs_seq[:, -self.ctx_size:]  # moving ctx
        new_action_seq = new_action_seq[:, -self.ctx_size:]
        return new_obs_seq, new_action_seq