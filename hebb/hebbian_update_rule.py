import torch
from enum import Enum


class HebbianUpdateMode(Enum):
    HebbianPCA = 1
    SoftWinnerTakesAll = 2
    SWTA_T = 3
    HPCA_T = 4
    CONTRASTIVE = 5


def soft_winner_takes_all(x, y, w, inverse_temeprature):
    r = (y * inverse_temeprature).softmax(dim=0)
    return torch.matmul(r, x) - r.sum(1, keepdim=True) * w


def hebbian_pca(x, y, w):
    out_channels = y.size()[0]
    mask = torch.arange(
        out_channels,
        device=x.device,
        dtype=x.dtype
    ).unsqueeze(0).repeat(
        out_channels,
        1
    ).le(
        torch.arange(out_channels, device=x.device, dtype=x.dtype).unsqueeze(1)
    ).to(dtype=x.dtype)

    return y.matmul(x) - (y.matmul(y.transpose(-2, -1)) * mask).matmul(w)
