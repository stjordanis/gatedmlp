import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialGatingUnit(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_seq: int,
        causal: bool = False,
        act=nn.Identity(),
        init_eps=1e-3,
    ):
        super().__init__()
        dim_out = dim // 2
        self.causal = causal

        self.norm = nn.LayerNorm(dim_out)
        self.proj = nn.Conv1d(dim_seq, dim_seq, 1)

        self.act = act

        init_eps /= dim_seq

        nn.init.uniform_(self.proj.weight, -init_eps, init_eps)
        nn.init.constant_(self.proj.bias, 1.0)

    def forward(self, x, gate_res=None):
        device, n = x.device, x.shape[1]

        res, gate = x.chunk(2, dim=-1)
        gate = self.norm(gate)

        weight, bias = self.proj.weight, self.proj.bias
        if self.causal:
            weight, bias = weight[:n, :n], bias[:n]
            mask = torch.ones(weight.shape[:2], device=device).triu_(1).bool()
            weight = weight.masked_fill(mask[..., None], 0.0)

        gate = F.conv1d(gate, weight, bias)

        if gate_res is not None:
            gate = gate + gate_res

        return self.act(gate) * res


class gMLPBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        seq_len: int,
        act=nn.GELU()
        # survival_prob: torch.Tensor
    ):
        super(gMLPBlock, self).__init__()

        self.norm = nn.LayerNorm(d_model)
        self.proj_1 = nn.Linear(d_model, d_ffn)
        self.activation = act
        self.spatial_gating_unit = SpatialGatingUnit(d_ffn, seq_len)
        self.proj_2 = nn.Linear(d_ffn // 2, d_model)
        # self.prob = survival_prob
        # self.m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob]))

    def forward(self, x):
        """
        if self.training and torch.equal(self.m.sample(), torch.zeros(1)):
            return x"""
        shorcut = x.clone()
        x = self.norm(x)
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shorcut
