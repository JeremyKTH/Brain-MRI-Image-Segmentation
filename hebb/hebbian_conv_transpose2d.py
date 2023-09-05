import torch
from hebb.hebbian_update_rule import SoftWinnerTakesAll

from hebb.unit_types import DotUnit


def to_2dvector(param, param_name):
    if type(param) == int:
        converted = (param, param)
    elif type(param) == list:
        if len(param) != 2:
            raise AttributeError(f'When {param_name} is list it should be of lenght 2 but it was {len(param)}')
        converted = tuple(param)
    elif type(param) == tuple:
        if (len(param)) != 2:
            raise AttributeError(f'When {param_name} is tuple it should be of length 2 but it was '
                                 f'{len(param)}')
        converted = param
    else:
        raise TypeError(f'{param_name} type {type(param)} is not supported. The supported types are'
                        f'int, list, and typle')

    return torch.tensor(converted)


class HebbianConvTranspose2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1,
                 bias=True,
                 dilation=1, padding_mode='zeros', device=None, dtype=None,
                 hebbian_mode=True,
                 unit_type=DotUnit(),
                 hebbian_update_rule=SoftWinnerTakesAll(0.02), patchwise=True):
        """
        Hebbian transposed convolution implemented in terms of dot product over sliding windows
        It can work in two modes:
         * Hebbian Mode (hebbian_mode=True). It will update weights according to hebbian rules determined by (mode) the
            updates happen automatically after each forward pass so no additional work necessary in the training loop
            apart from optimizer.step()

         * Standard Mode (hebbian_mode=False). In this mode it can use bias and standard backpropagation
        :param in_channels: Number of input channels (int)
        :param out_channels:  Number of output channels (int)
        :param kernel_size: Size of convolutional kernel (int or tuple)
        :param stride: Size of stride (int or tuple)
        :param padding: How many pixels to remove from result feature map (int or tuple)
        :param output_padding: Helps determine output shape (int or tuple)
        :param dilation: The kernel dilation (int or tuple)
        :param padding_mode: Only supports "zeros",
        :param device: Whether to keep the paramteres on a GPU accelerator
        :param dtype: Type of parameters
        :param hebbian_mode: Whether to work with hebbian updates
        :param hebbian_update_rule: Type of hebbian update. Update rules are defined in hebb.hebbian_update_rule
        :param patchwise: Whether to use patchsize update. Currently only patchsize update is implemented

        Usage in a train loop:

        model = HebbianConvTranspose2d(5, 4, 3)
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0)
        model.train()

        # Hebbian only update:
        for x, _ in data:
            model(x)
            model.local_update()
            opt.step()

        # Backpropagation only update
        for x, y in data:
            l = loss(model(x), y)
            loss.backward()
            opt.step()

        # Mixed update
        for x, y in data:
            l = loss(model(x), y)
            loss.backward()
            model.local_update()
            opt.step()

        """

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.__hebbian_mode = hebbian_mode

        if padding_mode != 'zeros':
            raise NotImplemented("Only padding mode zeros is supported")

        if groups != 1:
            raise NotImplementedError("Groups different from 1 not implemented")

        self.patchwise = patchwise
        self.__hebbian_update_rule = hebbian_update_rule

        self.__kernel_size = to_2dvector(kernel_size, 'kernel_size')
        self.__stride = to_2dvector(stride, 'stride')
        self.__dilation = to_2dvector(dilation, 'dilation')
        self.__padding = to_2dvector(padding, 'padding')
        self.__output_padding = to_2dvector(output_padding, 'output_padding')

        # TODO: Implement these
        if torch.equal(self.__output_padding, torch.tensor([1, 1])):
            raise NotImplementedError("Output Padding different from 1 not implemented")

        self.__weight = torch.nn.Parameter(
            data=torch.nn.init.xavier_normal_(
                torch.empty(
                    torch.Size([self.out_channels, self.in_channels*self.kernel_size[0]*self.kernel_size[1]]),
                    dtype=dtype,
                    device=device)
            ),
            requires_grad=True
        )

        self.__bias = torch.nn.Parameter(
            torch.zeros((out_channels, 1), dtype=dtype, device=device),
            requires_grad=True
        ) if bias else None

        self.__delta_w = torch.zeros_like(self.__weight)

        self.__upscale = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=in_channels,
            kernel_size=1,
            stride=self.stride,
            bias=False,
            dtype=dtype)
        self.__upscale.weight = torch.nn.Parameter(torch.ones_like(self.__upscale.weight), requires_grad=False)
        self.__unfold = torch.nn.Unfold(self.kernel_size,
                                        self.dilation,
                                        padding=tuple([k - 1 + (d-1) * (k-1) for (k, d) in zip(self.kernel_size,
                                                                                               self.dilation)]),
                                        stride=1)

        self.__unit_type = unit_type

    def __calc_output_size(self, input_size: tuple):
        # ignore output_padding
        return (torch.tensor(input_size) - 1) * self.__stride + \
                self.__dilation * (self.__kernel_size - 1) + self.__output_padding + 1

    def forward(self, x):
        # Get input shape and verify tensor dimensionality
        input_size = x.size()
        tensor_dim = len(input_size)
        if tensor_dim != 3 and tensor_dim != 4:
            raise RuntimeError(f'Expected 3D (unbatched) or 4D (batched) input to HebbianConvTransposed2D but got size'
                               f' {input_size}')
        if tensor_dim == 3:
            x = torch.unsqueeze(x, 0)

        # Calculate output shape for 3D and 4D tensors
        output_shape = torch.Size((
            x.size()[0],
            self.out_channels,
            *self.__calc_output_size(input_size[2:])
        ))

        # Evalueate transposed convolution
        unfolded_x = self.__unfold(self.__upscale(x))

        if self.hebbian_mode:
            # In hebbian mode bias is not used and delta_weights are updated according to hebbian rule
            unfolded_y = self.__unit_type(unfolded_x, self.__weight)

            if self.training:
                self.__update_delta_w(unfolded_x, unfolded_y)

            unpadded_result = unfolded_y.reshape(output_shape)
        else:
            # Backpropagation Mode implements a standard tranposed convolution
            unpadded_result = self.__unit_type(unfolded_x, self.__weight, self.__bias).reshape(output_shape)

        # Return padded result according to tensor shape and dimensionality
        # Probably there is a better way to do this
        if torch.all(self.__padding.eq(0)):
            return unpadded_result
        elif self.__padding[0] == 0:
            return unpadded_result[:, :, :, self.__padding[1]:-self.__padding[-1]]
        elif self.__padding[1] == 0:
            return unpadded_result[:, :, self.__padding[0]:-self.__padding[-0], :]
        else:
            return unpadded_result[:, :, self.__padding[0]:-self.__padding[-0], self.__padding[1]:-self.__padding[-1]]

    def __update_delta_w(self, x, y):
        if self.patchwise:
            self.__delta_w[:, :] += self.__hebbian_update_rule(x, y, self.__weight)
        else:
            raise NotImplementedError("Non-patchwise learning is not implemented")

    def local_update(self, alpha=1):
        if not self.__hebbian_mode:
            raise RuntimeError("Cannot do hebbian_update when not in hebbian_mode")

        if self.__weight.grad is None:
            self.__weight.grad = -alpha * self.__delta_w
        else:
            self.__weight.grad -= alpha * self.__delta_w

        self.__delta_w.zero_()

    # ----------------------------- #
    # Getters and Setters

    @property
    def kernel_size(self):
        return tuple(self.__kernel_size.tolist())

    @property
    def stride(self):
        return tuple(self.__stride.tolist())

    @property
    def dilation(self):
        return tuple(self.__dilation.tolist())

    @property
    def padding(self):
        return tuple(self.__padding.tolist())

    @property
    def output_padding(self):
        return tuple(self.__output_padding.tolist())

    @property
    def weight(self):
        # Convert internal representation of weights in format appropreate for ConvTranposed2d
        # The flip operations makes sure that a copy is returned
        return torch.nn.Parameter(
            self.__weight.data.reshape(
                self.out_channels,
                self.in_channels,
                torch.prod(torch.tensor(self.kernel_size))
            ).flip(
                [2]
            ).reshape(
                self.out_channels,
                self.in_channels,
                *self.kernel_size
            ).transpose(0, 1)
        )

    @weight.setter
    def weight(self, weight: torch.tensor):
        # When assigning weights expect the format appropreate for ConvTransposed2d and convert to internal
        # representation

        if type(weight) == torch.Tensor:
            weight = weight.clone()
        else:
            weight = torch.tensor(weight)

        # The flip operation causes copying
        self.__weight = torch.nn.Parameter(
            weight.transpose(0, 1).reshape(
                self.out_channels,
                self.in_channels,
                torch.prod(self.__kernel_size)
            ).flip(
                [2]
            ).reshape(self.out_channels, self.in_channels * torch.prod(self.__kernel_size)),

            requires_grad=True
        )

    @property
    def bias(self):
        if self.__bias is not None:
            return self.__bias.clone()
        return None

    @bias.setter
    def bias(self, b):
        self.__bias = torch.nn.Parameter(b, requires_grad=not self.hebbian_mode)

    @property
    def hebbian_mode(self):
        return self.__hebbian_mode

    @hebbian_mode.setter
    def hebbian_mode(self, mode: bool):
        self.__hebbian_mode = mode
        if self.__bias is not None:
            self.__bias.requires_grad = not mode
