import torch


def normalize_weight(w, dim):
    nrm_w = (w ** 2).sum(dim=dim, keepdim=True) ** 0.5
    nrm_w[nrm_w == 0] = 1.
    return w / nrm_w


class DotUnit:
    def __init__(self,
                 act=torch.nn.ReLU(inplace=True),
                 weight_normalize=True):
        self.__act = act
        self.__weight_normalize = weight_normalize

    def __call__(self, x, w, b=None):
        if self.__weight_normalize:
            w = normalize_weight(w, dim=1)
        if b is None:
            return self.__act(torch.matmul(w, x))

        return self.__act(torch.matmul(w, x) + b)


class RadialBasis:
    def __call__(self, x, w, bias=None):
        return torch.exp(torch.norm(x - w)**2)
