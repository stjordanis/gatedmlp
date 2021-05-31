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
