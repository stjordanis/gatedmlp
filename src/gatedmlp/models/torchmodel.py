import torch
import torch.nn as nn


class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn: int, seq_len: int):
        super().__init__()

        # Input Size = (*, d_ffn // 2)
        self.norm = nn.LayerNorm(d_ffn // 2)
        self.proj = nn.Linear(seq_len, seq_len)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = v.permute(0, 2, 1)
        v = self.proj(v)
        v = v.permute(0, 2, 1)
        return u * v


class gMLPBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        seq_len: int,
        # survival_prob: torch.Tensor
    ):
        super(gMLPBlock, self).__init__()

        self.norm = nn.LayerNorm(d_model)
        self.proj_1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.GELU()
        self.spatial_gating_unit = SpatialGatingUnit(d_ffn, seq_len)
        self.proj_2 = nn.Linear(d_ffn // 2, d_model)
        # self.prob = survival_prob
        # self.m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob]))

    def forward(self, x):
        if self.training and torch.equal(self.m.sample(), torch.zeros(1)):
            return x
        shorcut = x.clone()
        x = self.norm(x)
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shorcut
