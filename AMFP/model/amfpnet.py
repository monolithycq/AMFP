import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import Tensor, device
from typing import Union
import numpy as np
import math
from omegaconf import DictConfig
from AMFP.model.model_utils import Pooler_Head, Flatten_Head, ContrastiveWeight, AggregationRebuild, AutomaticWeightedLoss

class PE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, slots,observations,rtg=None):
        if rtg is not None:  # for encoder
            slots += self.pe[:, :slots.shape[1]]
            rtg += self.pe[:, slots.shape[1]:slots.shape[1] + rtg.shape[1]]
            observations += self.pe[:, slots.shape[1] + rtg.shape[1]: slots.shape[1] + rtg.shape[1] + observations.shape[1]]
            return slots, rtg, observations
        else:  # for decoder
            slots += self.pe[:, :slots.shape[1]]
            observations += self.pe[:, slots.shape[1]: slots.shape[1] + observations.shape[1]]
            return slots, observations

        #[bs, l, dim]
        # x += self.pe[:,:x.shape[1]]
        # return x



class AggSlotMAE(nn.Module):
    def __init__(self,  traj_len, dim, model_config):
        super(AggSlotMAE, self).__init__()
        self.embed_dim =model_config.embed_dim
        self.traj_len = traj_len
        self.dim = dim
        self.n_slots = model_config.n_slots
        self.slots = nn.Embedding(self.n_slots, self.embed_dim)
        self.pe_multi = PE(d_model=self.embed_dim)
        self.encoder_embed_dict = nn.ModuleDict()
        for key, shape in self.dim.items():
            self.encoder_embed_dict[key] = nn.Linear(shape, self.embed_dim)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                dim_feedforward=self.embed_dim * 4,
                nhead=model_config.n_head,
                dropout=model_config.pdrop,
                activation=F.gelu,
                norm_first=True,
                batch_first=True
            ), model_config.n_enc_layers, norm=nn.LayerNorm(self.embed_dim))

        self.pooler = Pooler_Head(self.n_slots, self.embed_dim, head_dropout=0.1)

        self.contrastive = ContrastiveWeight(model_config)
        self.aggregation = AggregationRebuild(model_config)

        self.obs_mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                dim_feedforward=self.embed_dim * 4,
                nhead=model_config.n_head,
                dropout=model_config.pdrop,
                activation=F.gelu,
                norm_first=True,
                batch_first=True
            ), model_config.n_enc_layers, norm=nn.LayerNorm(self.embed_dim))

        self.obs_decoder = nn.Linear(self.embed_dim, self.dim["states"])


        self.awl = AutomaticWeightedLoss(2)

    def decode(self,bottleneck,obs_mask):
        batch_size, s_len,_ = bottleneck.shape
        mask_obs = self.obs_mask_token.repeat(batch_size, obs_mask.shape[1], 1)
        b,o = self.pe_multi(bottleneck,mask_obs)
        dec_inputs = torch.cat([b, o], dim=1)
        decode_out = self.decoder(dec_inputs)

        obs_out = decode_out[:, s_len:]
        pred_o = self.obs_decoder(obs_out)
        return pred_o

    def slots_exact(self,obs,rtg,obs_mask):
        bs, len, _ = obs.shape
        slots = self.slots(torch.arange(self.n_slots, device=obs.device)).repeat(bs, 1, 1)

        rtg_emb = self.encoder_embed_dict['returns'](rtg)
        obs_emb = self.encoder_embed_dict['states'](obs)

        s, rtg, o = self.pe_multi(slots, obs_emb, rtg_emb)
        o_keep = o[obs_mask == 0].view(bs, -1, self.embed_dim)
        enc_inputs = torch.cat([s, rtg, o_keep], dim=1)

        encoded_keep = self.encoder(enc_inputs)
        bottleneck = encoded_keep[:, :slots.shape[1]]

        return bottleneck



    def valid(self,obs,obs_mask,rtg,padding_mask):
        bottleneck = self.slots_exact(obs,rtg,obs_mask)
        B, T = padding_mask.size()
        padding_mask = padding_mask.float()

        pred_o = self.decode(bottleneck,obs_mask)
        loss_o = F.mse_loss(pred_o, obs, reduction='none')
        loss_o *= 1 - padding_mask.view(B, T, -1)  # padding mask: 0 keep, 1 pad, so 1 - padding_mask needed
        loss_o = loss_o.mean() * (B * T) / (1 - padding_mask).sum()
        return loss_o




    def forward(self,obs_pn,obs_mask,rtg,pn,padding_mask):
        bs_pn, len, _ = obs_pn.shape
        slots = self.slots(torch.arange(self.n_slots, device=obs_pn.device)).repeat(bs_pn, 1, 1)
        rtg_emb = self.encoder_embed_dict['returns'](rtg)
        obs_emb = self.encoder_embed_dict['states'](obs_pn)

        s, rtg, o = self.pe_multi(slots, obs_emb, rtg_emb)
        o_keep = o[obs_mask == 0].view(bs_pn, -1, self.embed_dim)
        enc_inputs = torch.cat([s, rtg, o_keep], dim=1)

        encoded_pn = self.encoder(enc_inputs)

        bottleneck_pn = encoded_pn[:,:slots.shape[1]]

        p_bottleneck_pn = self.pooler(bottleneck_pn) # s_enc_out: [bs(1+pn), dimension]

        loss_cl, similarity_matrix, logits, index = self.contrastive(p_bottleneck_pn)
        rebuild_weight_matrix, agg_enc_out = self.aggregation(similarity_matrix, bottleneck_pn, index)  # agg_enc_out: [bs(1+pn) or bs,seq_len,d_model]

        oral_batch_size = bs_pn // (pn + 1)

        bottleneck_agg = agg_enc_out[:oral_batch_size]
        obs_mask_oral = obs_mask[:oral_batch_size]

        pred_o = self.decode(bottleneck_agg, obs_mask_oral)

        B, T = padding_mask.size()
        padding_mask = padding_mask.float()

        loss_o = F.mse_loss(pred_o, obs_pn[:oral_batch_size], reduction='none')
        loss_o *= 1 - padding_mask.view(B, T, -1)  # padding mask: 0 keep, 1 pad, so 1 - padding_mask needed
        loss_o = loss_o.mean() * (B * T) / (1 - padding_mask).sum()

        loss = self.awl(loss_cl, loss_o)

        return loss,loss_o,loss_cl


class AMFPNet(pl.LightningModule):
    def __init__(self,seed,env_name,ctx_size,future_horizon,stage,epochs,lr,obs_dim,return_dim,action_dim,return_type,model_config,**kwargs):
        super(AMFPNet, self).__init__()
        self.env_name = env_name
        self.ctx_size = ctx_size
        self.future_horizon = future_horizon
        self.traj_len = self.ctx_size+ self.future_horizon
        self.positive_nums = model_config.positive_nums
        self.dim = {'states': obs_dim, 'returns': return_dim, 'actions': action_dim}


        self.stage = stage  # task
        self.num_epochs = epochs
        self.lr = lr
        self.ar_mask_ratios = model_config.ar_mask_ratios
        self.rnd_mask_ratios = model_config.rnd_mask_ratios
        self.ar_mask_ratio_weights = model_config.ar_mask_ratio_weights

        self.model = AggSlotMAE(self.traj_len,self.dim,model_config)

        self.save_hyperparameters()

    def mask4cl(self,bs,len,mask_ratio,device):
        unmasked_len = max(1,int(len*(1-mask_ratio)))
        noise = torch.rand(size=(bs, len), device=device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep, 1 is mask
        mask = torch.ones([bs, len], device=device)
        mask[:, :unmasked_len] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask

    def forward(self,obs_pn,obs_mask,rtg,pn,padding_mask):
        return self.model.forward(obs_pn,obs_mask,rtg,pn,padding_mask)


    def training_step(self,batch,batch_idx):
        observations, actions, _return, valid_length, padding_mask = batch
        batch_size, _, _ = observations.shape
        o = observations[:, :self.traj_len]
        padding_mask = padding_mask[:, :self.traj_len]
        ar_mask_ratio = np.random.choice(self.ar_mask_ratios, 1, p=self.ar_mask_ratio_weights)[0]
        rnd_mask_ratio = np.random.choice(self.rnd_mask_ratios, 1)[0]
        pn = self.positive_nums

        keep_len = max(pn, int(self.ctx_size * (1 - ar_mask_ratio)))

        current_bs = batch_size*(1+pn)

        obs_pn = o.repeat(pn+1, 1, 1)

        history_mask = self.mask4cl(current_bs,keep_len,rnd_mask_ratio,observations.device)
        future_mask = torch.ones([current_bs, self.traj_len-keep_len], device=observations.device)
        obs_mask = torch.cat([history_mask, future_mask], dim=1)

        rtg = _return[:,keep_len-1:keep_len].repeat(pn+1, 1, 1)

        loss,loss_o,loss_cl = self(obs_pn,obs_mask,rtg,pn,padding_mask)

        self.log_dict({
            'train/train_loss': loss,
            'train/loss_o': loss_o,
            'train/loss_cl': loss_cl,
        },
            sync_dist=True)

        # return loss_cl
        return loss

    def validation_step(self,batch,batch_idx):
        observations, actions, _return, valid_length, padding_mask = batch
        batch_size, _, _ = observations.shape
        o = observations[:, :self.traj_len]
        padding_mask = padding_mask[:, :self.traj_len]
        ar_mask_ratio = np.random.choice(self.ar_mask_ratios, 1, p=self.ar_mask_ratio_weights)[0]
        rnd_mask_ratio = np.random.choice(self.rnd_mask_ratios, 1)[0]

        keep_len = max(1, int(self.ctx_size * (1 - ar_mask_ratio)))
        history_mask = self.mask4cl(batch_size, keep_len, rnd_mask_ratio, o.device)
        future_mask = torch.ones([batch_size, self.traj_len - keep_len], device=o.device)
        obs_mask = torch.cat([history_mask, future_mask], dim=1)
        rtg = _return[:,keep_len-1:keep_len]

        loss_o = self.model.valid(o,obs_mask,rtg,padding_mask)

        self.log_dict({
            'val/val_loss': loss_o,
        },
            sync_dist=True)



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
        }










    
