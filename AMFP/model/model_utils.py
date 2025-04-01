import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
# from typing import Any, Callable, Dict, Sequence, Tuple
# from collections import defaultdict
# import tqdm


class Flatten_Head(nn.Module):
    def __init__(self, seq_len, d_model, pred_len, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(seq_len*d_model, pred_len)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # [bs x n_vars x seq_len x d_model]
        x = self.flatten(x) # [bs x n_vars x (seq_len * d_model)]
        x = self.linear(x) # [bs x n_vars x seq_len]
        x = self.dropout(x) # [bs x n_vars x seq_len]
        return x

class Pooler_Head(nn.Module):
    def __init__(self, seq_len, d_model, head_dropout=0):
        super().__init__()

        pn = seq_len * d_model
        dimension = 128
        self.pooler = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(pn, pn // 2),
            nn.BatchNorm1d(pn // 2),
            nn.ReLU(),
            nn.Linear(pn // 2, dimension),
            nn.Dropout(head_dropout),
        )
    def forward(self, x):  # [(bs * n_vars) x seq_len x d_model]
        x = self.pooler(x) # [(bs * n_vars) x dimension]
        return x

class ContrastiveWeight(torch.nn.Module):

    def __init__(self, args):
        super(ContrastiveWeight, self).__init__()
        self.temperature = args.temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.kl = torch.nn.KLDivLoss(reduction='batchmean')
        self.positive_nums = args.positive_nums

        self.divide = args.divide   #xxxxxxx
        self.negative_nums = args.negative_nums

    def get_positive_and_negative_mask(self, similarity_matrix, cur_batch_size):
        diag = np.eye(cur_batch_size)
        mask = torch.from_numpy(diag)
        mask = mask.type(torch.bool)

        oral_batch_size = cur_batch_size // (self.positive_nums + 1)

        positives_mask = np.zeros(similarity_matrix.size())
        for i in range(self.positive_nums + 1):
            ll = np.eye(cur_batch_size, cur_batch_size, k=oral_batch_size * i)
            lr = np.eye(cur_batch_size, cur_batch_size, k=-oral_batch_size * i)
            positives_mask += ll
            positives_mask += lr

        positives_mask = torch.from_numpy(positives_mask).to(similarity_matrix.device)
        positives_mask[mask] = 0

        negatives_mask = 1 - positives_mask
        negatives_mask[mask] = 0

        return positives_mask.type(torch.bool), negatives_mask.type(torch.bool)

    def forward(self, batch_emb_om):
        cur_batch_shape = batch_emb_om.shape
        oral_batch_size = cur_batch_shape[0] // (self.positive_nums + 1)

        # get similarity matrix among mask samples
        norm_emb = F.normalize(batch_emb_om, dim=1)
        similarity_matrix = torch.matmul(norm_emb, norm_emb.transpose(0, 1))

        if self.divide == False:
            # get positives and negatives similarity
            positives_mask, negatives_mask = self.get_positive_and_negative_mask(similarity_matrix, cur_batch_shape[0])

            positives = similarity_matrix[positives_mask].view(cur_batch_shape[0], -1)
            negatives = similarity_matrix[negatives_mask].view(cur_batch_shape[0], -1)

            # generate predict and target probability distributions matrix
            logits = torch.cat((positives, negatives), dim=-1)
            y_true = torch.cat((torch.ones(cur_batch_shape[0], positives.shape[-1]), torch.zeros(cur_batch_shape[0], negatives.shape[-1])), dim=-1).to(batch_emb_om.device).float()

            index = []

        elif self.divide == True:
            whole_indices = torch.arange(cur_batch_shape[0]).to(norm_emb.device)
            negatives_indices = torch.arange(1, oral_batch_size + 1).unsqueeze(1).to(norm_emb.device) + torch.arange(self.negative_nums).to(norm_emb.device)
            positives_indices = whole_indices[None, oral_batch_size::oral_batch_size] + torch.arange(oral_batch_size)[:, None].to(norm_emb.device)
            index = torch.cat((positives_indices, negatives_indices), dim=-1)

            logits = similarity_matrix[torch.arange(oral_batch_size).unsqueeze(1), index]
            y_true = torch.cat((torch.ones(oral_batch_size, self.positive_nums), torch.zeros(oral_batch_size, self.negative_nums)), dim=-1).to(batch_emb_om.device).float()

        # else:
        #     index_list = []
        #     # get positives and negatives similarity
        #     negatives = torch.zeros(oral_batch_size, self.negative_nums, device=norm_emb.device)
        #     positives = torch.zeros(oral_batch_size, self.positive_nums, device=norm_emb.device)
        #
        #     for i in range(oral_batch_size):
        #         valid_indices = list(range(oral_batch_size))
        #         valid_indices.remove(i)
        #         selected_indices = np.random.choice(valid_indices, self.negative_nums, replace=False)
        #         negatives_indices = torch.tensor(selected_indices, device=norm_emb.device)
        #
        #         whole_indices = torch.arange(cur_batch_shape[0]).to(norm_emb.device)
        #         positives_indices = whole_indices[i+oral_batch_size::oral_batch_size]
        #
        #         indices = torch.cat((positives_indices,negatives_indices))
        #         index_list.append(indices)
        #
        #
        #         negatives[i,:] = similarity_matrix[i, negatives_indices]
        #         positives[i,:] = similarity_matrix[positives_indices,i]
        #
        #     logits = torch.cat((positives, negatives), dim=-1)
        #     y_true = torch.cat((torch.ones(oral_batch_size, positives.shape[-1]), torch.zeros(oral_batch_size, negatives.shape[-1])), dim=-1).to(batch_emb_om.device).float()
        #
        #     index = torch.stack(index_list)



        # multiple positives - KL divergence

        predict = self.log_softmax(logits / self.temperature)
        # predict = torch.clamp(predict, min=1e-8)
        loss = self.kl(predict, y_true)



        return loss, similarity_matrix, logits, index

class AggregationRebuild(torch.nn.Module):

    def __init__(self, args):
        super(AggregationRebuild, self).__init__()
        self.temperature = args.temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.positive_nums = args.positive_nums
        self.divide = args.divide  # xxxxxxx
        self.negative_nums = args.positive_nums

    def forward(self, similarity_matrix, batch_emb_om, index):

        cur_batch_shape = batch_emb_om.shape


        # get the weight among (oral, oral's masks, others, others' masks)
        similarity_matrix /= self.temperature
        if self.divide == False:
            similarity_matrix = similarity_matrix - torch.eye(cur_batch_shape[0]).to(similarity_matrix.device).float() * 1e12
            rebuild_weight_matrix = self.softmax(similarity_matrix)

            batch_emb_om = batch_emb_om.reshape(cur_batch_shape[0], -1)

            # generate the rebuilt batch embedding (oral, others, oral's masks, others' masks)
            rebuild_batch_emb = torch.matmul(rebuild_weight_matrix, batch_emb_om)
            # get oral' rebuilt batch embedding
            rebuild_oral_batch_emb = rebuild_batch_emb.reshape(cur_batch_shape[0], cur_batch_shape[1], -1)

        else:
            sub_similarity = similarity_matrix[torch.arange(index.shape[0]).unsqueeze(1),index]
            rebuild_weight_matrix = self.softmax(sub_similarity)

            neighbor_features = batch_emb_om[index]
            # rebuild_weight_matrix: (N, p+n) -> (N, P+n, 1, 1)
            # neighbor_features: (N, p+n, l, c)
            weighted_neighbors = rebuild_weight_matrix.unsqueeze(-1).unsqueeze(-1) * neighbor_features
            rebuild_oral_batch_emb = weighted_neighbors.sum(dim=1)
        return rebuild_weight_matrix, rebuild_oral_batch_emb

class AutomaticWeightedLoss(torch.nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum