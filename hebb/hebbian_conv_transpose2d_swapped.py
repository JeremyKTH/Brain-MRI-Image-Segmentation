import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from hebb.hebbian_conv_2d import HebbianConv2d


class HebbianConvTranspose2dSwapped(HebbianConv2d):
    """
    A 2d convolutional layer that learns through Hebbian plasticity
    """

    MODE_SWTA_T = 'swta_t'
    MODE_HPCA_T = 'hpca_t'
    MODE_CONTRASTIVE = 'contrastive'

    def __init__(self, in_channels, out_channels, kernel_size, stride, w_nrm=True, act=nn.ReLU(inplace=True),
                 mode=MODE_SWTA_T, k=0.02, patchwise=False, contrast=1., uniformity=False):
        """

        :param out_channels: output channels of the convolutional kernel
        :param in_channels: input channels of the convolutional kernel
        :param kernel_size: size of the convolutional kernel (int or tuple)
        :param stride: stride of the convolutional kernel (int or tuple
        :param w_nrm: whether to normalize the weight vectors before computing outputs
        :param act: the nonlinear activation function after convolution
        :param mode: the learning mode, either 'swta', 'swta_t, 'hpca', 'hpca_t', or 'contrastive'
        :param k: softmax inverse temperature parameter used for swta-type learning
        :param patchwise: whether updates for each convolutional patch should be computed separately,
        and then aggregated
        :param contrast: coefficient that rescales negative compared to positive updates in contrastive-type learning
        :param uniformity: whether to use uniformity weighting in contrastive-type learning.
        """

        super().__init__(out_channels, in_channels, kernel_size, stride, w_nrm, act, mode, k, patchwise)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.contrast = contrast
        self.uniformity = uniformity

    def apply_weights(self, x, w):
        """
        This function implements the logic that computes Hebbian plasticity rules from input x and output y. The
        resulting weight update is stored in buffer self.delta_w for later use.
        """

        return torch.conv_transpose2d(x, w, self.bias, self.stride)

    def update(self, x, y):
        """
        This function implements the logic that computes local plasticity rules from input x and output y. The
        resulting weight update is stored in buffer self.delta_w for later use.
        """

        if self.mode not in [self.MODE_SWTA, self.MODE_HPCA, self.MODE_SWTA_T, self.MODE_HPCA_T, self.MODE_CONTRASTIVE]:
            raise NotImplementedError(
                "Learning mode {} unavailable for {} layer".format(self.mode, self.__class__.__name__))

        if self.mode in [self.MODE_SWTA, self.MODE_HPCA]:
            # In case of swta-type or hpca-type learning, use the learning rules for ordinary convolution,
            # but exchanging x and y
            super().update(y, x)

        if self.mode == self.MODE_SWTA_T:
            # Logic for swta-type learning in transpose convolutional layers
            r = (y * self.k).softmax(dim=1)
            r = F.unfold(r, kernel_size=self.kernel_size, stride=self.stride)
            r = r.permute(0, 2, 1).reshape(-1, self.out_channels, self.kernel_size[0] * self.kernel_size[1]).permute(2,
                                                                                                                     1,
                                                                                                                     0)
            dec = r.sum(2, keepdim=True) * self.weight.permute(2, 3, 1, 0).reshape(-1, self.out_channels,
                                                                                   self.in_channels)
            if self.patchwise: dec = dec.sum(dim=0, keepdim=True)
            self.delta_w[:, :, :, :] = (r.matmul(x.permute(0, 2, 3, 1).reshape(1, -1, x.size(1))) - dec).permute(2, 1,
                                                                                                                 0).reshape_as(
                self.weight)

        if self.mode == self.MODE_HPCA_T:
            # Logic for hpca-type learning in transpose convolutional layers
            r = y
            r = F.unfold(r, kernel_size=self.kernel_size, stride=self.stride)
            r = r.permute(0, 2, 1).reshape(-1, self.out_channels, self.kernel_size[0] * self.kernel_size[1]).permute(2,
                                                                                                                     1,
                                                                                                                     0)
            l = (torch.arange(self.out_channels, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(self.out_channels,
                                                                                                     1) <= torch.arange(
                self.out_channels, device=x.device, dtype=x.dtype).unsqueeze(1)).to(dtype=x.dtype)
            dec = (r.matmul(r.transpose(-2, -1)) * l.unsqueeze(0)).matmul(
                self.weight.permute(2, 3, 1, 0).reshape(-1, self.out_channels, self.in_channels))
            if self.patchwise: dec = dec.sum(dim=0, keepdim=True)
            self.delta_w[:, :, :, :] = (r.matmul(x.permute(0, 2, 3, 1).reshape(1, -1, x.size(1))) - dec).permute(2, 1,
                                                                                                                 0).reshape_as(
                self.weight)

        if self.mode == self.MODE_CONTRASTIVE:
            y_norm = (y ** 2).sum(dim=1, keepdim=True) ** 0.5
            y_norm[y_norm == 0] = 1.
            y = y / y_norm
            y_unf = F.unfold(y, _pair(3), padding=_pair(1))
            y_unf = y_unf.permute(0, 2, 1).reshape(y_unf.size(0), y_unf.size(2), y.size(1), 9)

            x_norm = (x ** 2).sum(dim=1, keepdim=True) ** 0.5
            x_norm[x_norm == 0] = 1.
            x = x / x_norm
            x_unf = F.unfold(x, _pair(3), padding=_pair(1))
            x_unf = x_unf.permute(0, 2, 1).reshape(x_unf.size(0), x_unf.size(2), x.size(1), 9)

            L = - y_unf.sum(-1).reshape(-1, y.size(1)) * y.permute(0, 2, 3, 1).reshape(-1, y.size(1))
            if self.uniformity:
                uniformity_map = (
                            x_unf.sum(-1).reshape(-1, x.size(1)) * x.permute(0, 2, 3, 1).reshape(-1, x.size(1))).sum(
                    dim=-1, keepdim=True)
                L *= torch.conv_transpose2d(uniformity_map.reshape(x.size(0), 1, x.size(1), x.size(2)),
                                            torch.ones([1, 1, *self.kernel_size], device=x.device, dtype=x.dtype),
                                            self.stride).reshape(-1, 1)
            idx = torch.randperm(y_unf.size(0), device=y_unf.device, dtype=y_unf.dtype)
            L += self.contrast * y_unf[idx].sum(-1).reshape(-1, y.size(1)) * y.permute(0, 2, 3, 1).reshape(-1,
                                                                                                           y.size(1))
            L = L.sum()
            self.zero_grad()
            L.backward()
            self.delta_w[:, :, :, :] = self.weight.grad.clone().detach()
            self.zero_grad()
