import torch


class SoftWinnerTakesAll:
    def __init__(self, inverse_temperature):
        self.__inverse_temperature = inverse_temperature

    def __call__(self, x, y, w):
        r = (y * self.__inverse_temperature).softmax(dim=-2)
        #return torch.einsum('bjn,bin->ji', r, x) - r.sum(dim=(0, 2)).unsqueeze(1) * w
        # einsum adds an additional overhead (~10%) --> matmul is more efficient
        return r.transpose(1, 2).reshape(r.shape[1], -1).matmul(x.transpose(0, 1).reshape(-1, x.shape[1])) - r.sum(dim=(0, 2)).unsqueeze(1) * w


class HebbianPCA:
    def __call__(self, x, y, w):
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
