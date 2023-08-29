import torch


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
                 stride=1, padding=0, output_padding=0, groups=1, bias=False,
                 dilation=1, padding_mode='zeros', device=None, dtype=None):
        super().__init__()
        # TODO: Add hebbian learning
        # TODO: Add padding
        self.in_channels = in_channels
        self.out_channels = out_channels

        if padding_mode != 'zeros':
            raise NotImplementedError("Only padding mode zeros is supported")

        if bias:
            raise NotImplementedError("Biases are not supported with hebbian learning")

        if groups != 1:
            raise NotImplementedError("Groups different from 1 not implemented")

        self.__kernel_size = to_2dvector(kernel_size, 'kernel_size')
        self.__stride = to_2dvector(stride, 'stride')
        self.__dilation = to_2dvector(dilation, 'dilation')
        self.__padding = to_2dvector(padding, 'padding')
        self.__output_padding = to_2dvector(output_padding, 'output_padding')

        # TODO: Implement these
        if torch.equal(self.__output_padding, torch.tensor([1, 1])):
            raise NotImplementedError("Output Padding different from 1 not implemented")

        self.__weight = torch.nn.init.xavier_normal_(
            torch.empty(
                torch.Size([self.out_channels, self.in_channels*self.kernel_size[0]*self.kernel_size[1]]),
                dtype=dtype,
                device=device,
                requires_grad=True))

        self.__upscale = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=in_channels,
            kernel_size=1,
            stride=self.stride,
            bias=False,
            dtype=dtype)
        self.__upscale.weight = torch.nn.Parameter(torch.ones_like(self.__upscale.weight))
        self.__unfold = torch.nn.Unfold(self.kernel_size,
                                        self.dilation,
                                        padding=tuple([k - 1 + (d-1) * (k-1) for (k, d) in zip(self.kernel_size,
                                                                                               self.dilation)]),
                                        stride=1)

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

        # Calculate output shape for 3D and 4D tensors
        output_shape = torch.Size((
            input_size[0],
            self.out_channels,
            *self.__calc_output_size(input_size[2:]))) if tensor_dim == 4 else torch.Size((
                self.out_channels,
                *self.__calc_output_size(input_size[2:])))

        # Evalueate transposed convolution
        unpadded_result = torch.matmul(
                self.__weight,
                self.__unfold(self.__upscale(x))
            ).reshape(output_shape)

        # Return padded result according to tensor shape and dimensionality
        # Probably there is a better way to do this
        if torch.all(self.__padding.eq(0)):
            return unpadded_result
        elif self.__padding[0] == 0:
            if tensor_dim == 3:
                return unpadded_result[:, :, self.__padding[1]:-self.__padding[-1]]
            else:
                return unpadded_result[:, :, :, self.__padding[1]:-self.__padding[-1]]
        elif self.__padding[1] == 0:
            if tensor_dim == 3:
                return unpadded_result[:, self.__padding[0]:-self.__padding[-0], :]
            else:
                return unpadded_result[:, :, self.__padding[0]:-self.__padding[-0], :]
        else:
            if tensor_dim == 3:
                return unpadded_result[:, self.__padding[0]:-self.__padding[-0],
                                       self.__padding[1]:-self.__padding[-1]]
            else:
                return unpadded_result[:, :, self.__padding[0]:-self.__padding[-0],
                                       self.__padding[1]:-self.__padding[-1]]

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
        # TODO: Clone the weight before returning
        # Convert internal representation of weights in format appropreate for ConvTranposed2d
        return torch.nn.Parameter(
            self.__weight.reshape(
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

        self.__weight = weight.transpose(0, 1).reshape(
            self.out_channels,
            self.in_channels,
            torch.prod(self.__kernel_size)
        ).flip(
            [2]
        ).reshape(self.out_channels, self.in_channels * torch.prod(self.__kernel_size))
