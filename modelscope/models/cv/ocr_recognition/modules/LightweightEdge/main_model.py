# Copyright (c) Alibaba, Inc. and its affiliates.

from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision


class LightweightEdge(nn.Module):
    """
        基于混合rep block的nas模型
        Args:
            input (tensor): batch of input images
    """

    def __init__(self):
        super(LightweightEdge, self).__init__()
        self.model = torchvision.models.mobilenet_v3_small(
            weights=torchvision.models.mobilenetv3.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        ).features
        self.embed_dim = 96
        self.head = nn.Linear(self.embed_dim, 6625)

    def forward(self, input):
        # RGB2GRAY
        b = input.size(0)
        x = self.model(input)
        x = x.reshape(b, -1, self.embed_dim)
        return self.head(x)
