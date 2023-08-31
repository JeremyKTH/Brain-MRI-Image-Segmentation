import torch
import torch.nn as nn
import torch.nn.functional as f

from hebb.hebbian_update_rule import soft_winner_takes_all
from hebb.hebbian_update_rule import hebbian_pca
from hebb.hebbian_update_rule import HebbianUpdateMode
from hebb.hebbian_update_rule import normalize_weight


class HebbianConv2d(nn.Module):
    """
    A 2d convolutional layer that learns through Hebbian plasticity
    """

    def __init__(self, out_channels, in_channels, kernel_size, stride,
                 w_nrm=True, act=nn.ReLU(inplace=True),
                 mode=HebbianUpdateMode.SoftWinnerTakesAll, k=0.02, patchwise=True):
        """

        :param out_channels: output channels of the convolutional kernel
        :param in_channels: input channels of the convolutional kernel
        :param kernel_size: size of the convolutional kernel (int or tuple)
        :param stride: stride of the convolutional kernel (int or tuple
        :param w_nrm: whether to normalize the weight vectors before computing outputs
        :param act: the nonlinear activation function after convolution
        :param mode: the learning mode, either 'swta' or 'hpca'
        :param k: softmax inverse temperature parameter used for swta-type learning
        :param patchwise: whether updates for each convolutional patch should be computed separately,
        and then aggregated

        """

        super().__init__()
        self.mode = mode
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = (kernel_size, kernel_size) if type(kernel_size) == int else tuple(kernel_size)
        self.stride = stride

        self.weight = nn.Parameter(torch.empty((out_channels, in_channels, *kernel_size)), requires_grad=True)
        nn.init.xavier_normal_(self.weight)
        self.w_nrm = w_nrm
        self.bias = nn.Parameter(torch.zeros(self.out_channels), requires_grad=True)
        self.act = act
        self.register_buffer('delta_w', torch.empty_like(self.weight))

        self.k = k
        self.patchwise = patchwise

    def apply_weights(self, x, w):
        """
        This function provides the logic for combining input x and weight w
        """

        return torch.conv2d(x, w, self.bias, self.stride)

    def forward(self, x):
        w = normalize_weight(self.weight, dim=(1, 2, 3)) if self.w_nrm else self.weight

        y = self.act(self.apply_weights(x, w))
        if self.training:
            self.update(x, y)
        return y

    def update(self, x, y):
        """
        This function implements the logic that computes local plasticity rules from input x and output y. The
        resulting weight update is stored in buffer self.delta_w for later use.
        """

        if self.mode not in [HebbianUpdateMode.HebbianPCA, HebbianUpdateMode.SoftWinnerTakesAll]:
            raise NotImplementedError(f'Learning mode {self.mode} unavailable for layer {self.__class__.__name__}')

        # The result of this reshape is a 2D matrix with dimensions: (batch_size * patches, kernel_size^2*channels)
        x_unf = f.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        x_unf = x_unf.permute(0, 2, 1).reshape(-1, x_unf.size(1))
        y_unf = y.permute(1, 0, 2, 3).reshape(self.out_channels, -1)

        if self.mode == HebbianUpdateMode.SoftWinnerTakesAll:
            # Logic for swta-type learning
            if self.patchwise:
                self.delta_w[:, :, :, :] = soft_winner_takes_all(
                    x_unf,
                    y_unf,
                    self.k,
                    self.weight.reshape(self.out_channels, -1)
                ).reshape_as(self.weight)
            else:
                raise NotImplementedError("Non-patchwise learning is not implemented for SWTA ")
                # r = (y * self.k).softmax(dim=1).permute(2, 3, 1, 0)
                # krn = torch.eye(len(self.weight[0]), device=x.device, dtype=x.dtype).view(len(self.weight[0]),
                #                                                                           self.in_channels,
                #                                                                           *self.kernel_size)
                # dec = torch.conv_transpose2d(
                #     (
                #             r.sum(dim=-1, keepdim=True) * self.weight.reshape(1, 1, self.out_channels, -1)
                #      ).permute(2, 3, 0, 1),
                #     krn,
                #     self.stride
                # )
                #
                # self.delta_w[:, :, :, :] = (
                #         r.permute(2, 3, 0, 1).reshape(
                #             self.out_channels,
                #             -1
                #         ).matmul(x_unf) - f.unfold(
                #                             dec,
                #                             kernel_size=self.kernel_size,
                #                             stride=self.stride).sum(dim=-1)
                # ).reshape_as(self.weight)

        if self.mode == HebbianUpdateMode.HebbianPCA:
            # Logic for hpca-type learning
            if self.patchwise:
                self.delta_w[:, :, :, :] = hebbian_pca(
                    x_unf,
                    y_unf,
                    self.weight.view(self.out_channels, -1)
                ).reshape_as(self.weight)
            else:
                raise NotImplementedError("Non-patchwise learning is not implemented for HPCA")
                # r = y.permute(2, 3, 1, 0)
                # l = (torch.arange(self.out_channels, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(
                #     self.out_channels, 1) <= torch.arange(self.out_channels, device=x.device, dtype=x.dtype
                #     ).unsqueeze(
                #     1)).to(dtype=x.dtype)
                # dec = torch.conv_transpose2d(
                #     (r.matmul(r.transpose(-2, -1)) * l.unsqueeze(0).unsqueeze(1)).permute(3, 2, 0, 1), self.weight,
                #     self.stride)
                #
                # self.delta_w[:, :, :, :] = (
                #         r.permute(2, 3, 0, 1).reshape(
                #                 self.out_channels,
                #                 -1
                #             ).matmul(x_unf) - f.unfold(
                #                                 dec,
                #                                 kernel_size=self.kernel_size,
                #                                 stride=self.stride).sum(dim=-1)
                # ).reshape_as(self.weight)

    def local_update(self, alpha=1):
        """
        This function transfers a previously computed weight update, stored in buffer self.delta_w, to the gradient
        self.weight.grad of the weigth parameter.

        This function should be called before optimizer.step(), so that the optimizer will use the locally computed
        update as optimization direction. Local updates can also be combined with end-to-end updates by calling this
        function between loss.backward() and optimizer.step(). loss.backward will store the end-to-end gradient in
        self.weight.grad, and this function combines this value with self.delta_w as
        self.weight.grad = self.weight.grad - alpha * self.delta_w
        Parameter alpha determines the scale of the local update compared to the end-to-end gradient in the combination.

        """

        if self.weight.grad is None:
            self.weight.grad = -alpha * self.delta_w
        else:
            self.weight.grad -= alpha * self.delta_w
        self.delta_w.zero_()
