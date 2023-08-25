import unittest
from hebb import HebbianConvTranspose2d
import torch


class TestHebbianConvTranpose2d(unittest.TestCase):
    def test_kernel_size_conversion(self):
        heb_conv = HebbianConvTranspose2d(5, 5, 3)
        self.assertEqual(heb_conv.kernel_size, (3, 3))

        heb_conv = HebbianConvTranspose2d(5, 5, [5, 7])
        self.assertEqual(heb_conv.kernel_size, (5, 7))

        heb_conv = HebbianConvTranspose2d(5, 5, (4, 6))
        self.assertEqual(heb_conv.kernel_size, (4, 6))

        try:
            HebbianConvTranspose2d(5, 5, [2, 3, 4])
            self.assertTrue(False, "HebbianConvTranpose2d should throw an exception of type Attribute error"
                            "for kernel sizes longer than 2 dimension")
        except AttributeError:
            self.assertTrue(True)

    def test_stride_conversion(self):
        heb_conv = HebbianConvTranspose2d(5, 5, 3, stride=3)
        self.assertEqual(heb_conv.stride, (3, 3))

        heb_conv = HebbianConvTranspose2d(5, 5, 3, stride=[5, 7])
        self.assertEqual(heb_conv.stride, (5, 7))

        heb_conv = HebbianConvTranspose2d(5, 5, 3, stride=(4, 6))
        self.assertEqual(heb_conv.stride, (4, 6))

        try:
            HebbianConvTranspose2d(5, 5, 3, stride=[2, 3, 4])
            self.assertTrue(False, "HebbianConvTranpose2d should throw an exception of type Attribute error"
                            "for stride sizes longer than 2 dimension")
        except AttributeError:
            self.assertTrue(True)

    def test_dilation_conversion(self):
        heb_conv = HebbianConvTranspose2d(5, 5, 3, dilation=3)
        self.assertEqual(heb_conv.dilation, (3, 3))

        heb_conv = HebbianConvTranspose2d(5, 5, 3, dilation=[5, 7])
        self.assertEqual(heb_conv.dilation, (5, 7))

        heb_conv = HebbianConvTranspose2d(5, 5, 3, dilation=(4, 6))
        self.assertEqual(heb_conv.dilation, (4, 6))

        try:
            HebbianConvTranspose2d(5, 5, 3, dilation=[2, 3, 4])
            self.assertTrue(False, "HebbianConvTranpose2d should throw an exception of type Attribute error"
                                   "for dilation sizes longer than 2 dimension")
        except AttributeError:
            self.assertTrue(True)

    # def test_padding_conversion(self):
    #     heb_conv = HebbianConvTranspose2d(5, 5, 3, padding=3)
    #     self.assertEqual(heb_conv.padding, (3, 3))
    #
    #     heb_conv = HebbianConvTranspose2d(5, 5, 3, padding=[5, 7])
    #     self.assertEqual(heb_conv.padding, (5, 7))
    #
    #     heb_conv = HebbianConvTranspose2d(5, 5, 3, padding=(4, 6))
    #     self.assertEqual(heb_conv.padding, (4, 6))
    #
    #     try:
    #         HebbianConvTranspose2d(5, 5, 3, padding=[2, 3, 4])
    #         self.assertTrue(False, "HebbianConvTranpose2d should throw an exception of type Attribute error"
    #                                "for padding sizes longer than 2 dimension")
    #     except AttributeError:
    #         self.assertTrue(True)
    #
    #
    # def test_output_padding_conversion(self):
    #     heb_conv = HebbianConvTranspose2d(5, 5, 3, output_padding=3)
    #     self.assertEqual(heb_conv.output_padding, (3, 3))
    #
    #     heb_conv = HebbianConvTranspose2d(5, 5, 3, output_padding=[5, 7])
    #     self.assertEqual(heb_conv.output_padding, (5, 7))
    #
    #     heb_conv = HebbianConvTranspose2d(5, 5, 3, output_padding=(4, 6))
    #     self.assertEqual(heb_conv.output_padding, (4, 6))
    #
    #     try:
    #         HebbianConvTranspose2d(5, 5, 3, output_padding=[2, 3, 4])
    #         self.assertTrue(False, "HebbianConvTranpose2d should throw an exception of type Attribute error"
    #                                "for output_padding sizes longer than 2 dimension")
    #     except AttributeError:
    #         self.assertTrue(True)

    def test_kernel_shape(self):
        for in_channels in range(1, 3):
            for out_channels in range(1, 3):
                for kernel_size in range(1, 5):
                    heb_conv = HebbianConvTranspose2d(in_channels, out_channels, kernel_size)
                    conv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
                    self.assertEqual(heb_conv.weight.size(), conv.weight.size())

    def test_weight_getter(self):
        for in_channels in range(1, 3):
            for out_channels in range(1, 3):
                for kernel_size in range(1, 5):
                    heb_conv = HebbianConvTranspose2d(in_channels, out_channels, kernel_size)
                    conv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
                    conv.weight = heb_conv.weight

                    self.assertTrue(torch.equal(conv.weight, heb_conv.weight), "weight getter doesn't work")

    def test_weight_setter(self):
        for in_channels in range(1, 3):
            for out_channels in range(1, 3):
                for kernel_size in range(1, 5):
                    heb_conv = HebbianConvTranspose2d(in_channels, out_channels, kernel_size)
                    conv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
                    heb_conv.weight = conv.weight.data
                    self.assertTrue(torch.equal(conv.weight, heb_conv.weight), "weight setter doesn't work")

    def test_forward(self):
        for in_channels in range(1, 3):
            for out_channels in range(1, 3):
                for kernel_size in range(1, 5):
                    heb_conv = HebbianConvTranspose2d(in_channels, out_channels, kernel_size)
                    conv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, bias=False)
                    conv.weight = heb_conv.weight
                    inp = torch.randn(1, in_channels, 10, 10)

                    # Numerical issues can casue problems when directly comparing
                    self.assertTrue(torch.all(torch.abs(heb_conv(inp) - conv(inp)) < 0.000001))

    def test_forward_stride(self):
        for in_channels in range(1, 3):
            for out_channels in range(1, 3):
                for kernel_size in range(1, 5):
                    heb_conv = HebbianConvTranspose2d(in_channels, out_channels, kernel_size, stride=2)
                    conv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, bias=False, stride=2)
                    conv.weight = heb_conv.weight
                    inp = torch.randn(1, in_channels, 10, 10)

                    # Numerical issues can casue problems when directly comparing
                    self.assertTrue(torch.all(torch.abs(heb_conv(inp) - conv(inp)) < 0.000001))

    def test_forward_dilation(self):
        for in_channels in range(1, 3):
            for out_channels in range(1, 3):
                for kernel_size in range(1, 5):
                    heb_conv = HebbianConvTranspose2d(in_channels, out_channels, kernel_size, dilation=2)
                    conv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, bias=False, dilation=2)
                    conv.weight = heb_conv.weight
                    inp = torch.randn(1, in_channels, 10, 10)

                    # Numerical issues can casue problems when directly comparing
                    self.assertTrue(torch.all(torch.abs(heb_conv(inp) - conv(inp)) < 0.000001))
