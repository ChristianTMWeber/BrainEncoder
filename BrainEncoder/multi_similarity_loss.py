# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import torch
from torch import nn

class MultiSimilarityLoss(nn.Module):
    def __init__(self):#, cfg):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1

        self.scale_pos = 2.0  # cfg.LOSSES.MULTI_SIMILARITY_LOSS.SCALE_POS
        self.scale_neg = 40.0 # cfg.LOSSES.MULTI_SIMILARITY_LOSS.SCALE_NEG


    def SqEuclideanDistance(self, x, y):
        """
        Compute the Euclidean Distance between two tensors
        """
        return torch.pow(x - y, 2).sum(1)

    def forward(self, features, labels, maxDistForPosPair=0, minDistForNegativePair=0, nUnaugmentedSamples = -1):
        assert features.size(0) == labels.size(0), \
            f"features.size(0): {features.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = features.size(0)
        sim_mat = torch.matmul(features, torch.t(features))

        epsilon = 1e-5
        loss = list()

        for i in range(batch_size):
            if i == nUnaugmentedSamples: break

            sq_dist = self.SqEuclideanDistance(labels[i], labels)

            pos_pair_ = sim_mat[i][sq_dist <= (maxDistForPosPair**2 + epsilon) ]
            #pos_pair_ = sim_mat[i][labels == labels[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]

            if len(pos_pair_) < 1: continue

            neg_pair_ = sim_mat[i][sq_dist > (minDistForNegativePair**2 + epsilon) ]
            #neg_pair_ = sim_mat[i][labels != labels[i]]

            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]

            if len(neg_pair_) < 1: continue
            pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            # weighting step
            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss) #/ batch_size
        return loss
