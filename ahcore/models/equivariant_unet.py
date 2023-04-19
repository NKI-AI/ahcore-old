# encoding: utf-8
"""Equivariant u-net as ..."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from escnn import gspaces  # type: ignore
from escnn import nn as enn  # type: ignore


def double_conv(
        group: gspaces.r2.general_r2.GeneralOnR2,
        in_planes: int,
        out_planes: int,
        stride: int = 1,
        dropout_rate: float = 0.3,
        activation=enn.ReLU,
):
    """Basic double convolutional layer with activation and batch normalization."""
    # TODO: Bias is set to false but check how that alligns with normal torch implementation...

    # Define input and output field type
    in_type = enn.FieldType(group, in_planes * [group.regular_repr])
    out_type = enn.FieldType(group, out_planes * [group.regular_repr])

    return enn.SequentialModule(
        enn.R2Conv(in_type, out_type, kernel_size=3, stride=stride, padding=1, bias=False),
        enn.PointwiseDropout(out_type, dropout_rate, inplace=False),
        enn.InnerBatchNorm(out_type),
        activation(out_type),
        enn.R2Conv(out_type, out_type, kernel_size=3, stride=stride, padding=1, bias=False),
        enn.PointwiseDropout(out_type, dropout_rate, inplace=False),
        enn.InnerBatchNorm(out_type),
        # pwdw seperable conv
        enn.R2Conv(out_type, out_type, kernel_size=3, stride=stride, padding=1, groups=out_planes, bias=False),
        activation(out_type),
    )


def unet_downsample_layer(group, in_planes, out_planes, activation=enn.ReLU):
    """
    Basic UNet downsample layer with consisting of double convolutional layers followed
    with a 2d max pooling layer wrapped in nn.Sequential.
    # TODO: added point max-pool here because of unet implementation tryout different ones!.
    """
    in_type = enn.FieldType(group, in_planes * [group.regular_repr])
    return enn.SequentialModule(
        enn.PointwiseMaxPool(in_type, kernel_size=2), double_conv(group, in_planes, out_planes, activation=activation)
    )


class UnetUpsampleLayer(nn.Module):
    """
    Upsampling (by either bilinear interpolation or transpose convolutions) followed by concatenation of feature
    map from contracting path, followed by DoubleConv.
    """

    def __init__(self, group, in_planes: int, out_planes: int, bilinear: bool = False):
        super().__init__()

        self.group = group
        self.in_type = enn.FieldType(group, in_planes * [group.regular_repr])
        self.in_type_div2 = enn.FieldType(group, (in_planes // 2) * [group.regular_repr])

        self.upsample = None
        if bilinear:
            self.upsample = nn.Sequential(
                enn.R2Upsampling(self.in_type, scale_factor=2, mode="bilinear", align_corners=True),
                enn.R2Conv(self.in_type, self.in_type_div2, kernel_size=1, bias=False),
            )
        else:
            self.upsample = enn.R2ConvTransposed(self.in_type, self.in_type_div2, kernel_size=2, stride=2, bias=False)

        self.conv = double_conv(group, in_planes, out_planes)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        # Get torch tensors to pad and concatenate.
        x1_tensor = x1.tensor
        x2_tensor = x2.tensor

        # Pad x1 to the size of x2.
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]
        x1_tensor = F.pad(x1_tensor, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        # Concatenate along the channels axis.
        cat_tensor = torch.cat([x2_tensor, x1_tensor], dim=1)

        # Wrap in geomteric tensor.
        geom_cat_tensor = enn.GeometricTensor(cat_tensor, type=self.in_type)

        # Apply double conv.
        return self.conv(geom_cat_tensor)


class E2UNet(nn.Module):
    """
    Paper: `U-Net: Convolutional Networks for Biomedical Image Segmentation
    <https://arxiv.org/abs/1505.04597>`_
    Paper authors: Olaf Ronneberger, Philipp Fischer, Thomas Brox
    Args:
        num_classes     : Number of output classes required
        input_channels  : Number of channels in input images (default 3)
        num_layers      : Number of layers in each side of U-net (default 5)
        hidden_features : Number of features in first layer (default 64)
        bilinear        : Whether to use bilinear interpolation or transposed convolutions (default) for upsampling.
    """

    def __init__(
            self,
            group,
            num_classes: int,
            input_channels: int = 3,
            num_layers: int = 5,
            hidden_features: int = 64,
            bilinear: bool = False,
            apply_softmax_out: bool = False,
            return_features: bool = False,
    ):

        # Check wether num_layers is more than zero.
        if num_layers < 1:
            raise ValueError(f"num_layers = {num_layers}, expected: num_layers > 0")

        super().__init__()
        self.group = group
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.hidden_features = hidden_features
        self.bilinear = bilinear
        self.apply_softmax_out = apply_softmax_out
        self.return_features = return_features

        # Define a preprocessing conv to go from trivial to regular representation.
        self.feat_type_in_triv = enn.FieldType(group, self.input_channels * [group.trivial_repr])
        feat_type_out_triv = enn.FieldType(group, 3 * [group.regular_repr])
        self.preproc_conv = enn.R2Conv(
            self.feat_type_in_triv, feat_type_out_triv, kernel_size=3, padding=1, bias=False
        )

        # Create layers of the UNet model.
        self.layers = self.create_unet()

    def create_unet(self):
        group = self.group

        # Define a feature extractor.
        layers = [double_conv(group, self.input_channels, self.hidden_features)]

        # Define the down path.
        feats = self.hidden_features
        for _ in range(self.num_layers - 1):
            layers.append(unet_downsample_layer(group, feats, feats * 2))
            feats *= 2

        # Define the up path.
        for _ in range(self.num_layers - 1):
            layers.append(UnetUpsampleLayer(group, feats, feats // 2, self.bilinear))
            feats //= 2

        # Define the final classification layer.
        classify_in_type = enn.FieldType(group, feats * [group.regular_repr])
        output_type = enn.FieldType(group, self.num_classes * [group.trivial_repr])
        layers.append(enn.R2Conv(classify_in_type, output_type, kernel_size=1, bias=False))
        return nn.ModuleList(layers)

    def forward(self, input_data) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor]]:
        # Wrap in tensor and map to regular representation.
        input_data = enn.GeometricTensor(input_data, self.feat_type_in_triv)
        input_data = self.preproc_conv(input_data)

        # Feature extraction
        xi = [self.layers[0](input_data)]

        # Down path
        for layer in self.layers[1: self.num_layers]:
            xi.append(layer(xi[-1]))

        # Up path
        for i, layer in enumerate(self.layers[self.num_layers: -1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])

        # Final classification layer
        output = self.layers[-1](xi[-1])

        # Unwrap geomteric tensor.
        output = output.tensor

        # Apply softmax to output distribution over classes.
        if self.apply_softmax_out:
            output = torch.softmax(output, dim=1)

        # Return features if required.
        if self.return_features:
            return output, tuple(xi)
        return output


if __name__ == "__main__":
    # Define the group.
    group = gspaces.rot2dOnR2(N=8)

    # Define the model.
    model = E2UNet(
        group=group,
        num_classes=3,
        input_channels=3,
        num_layers=3,
        hidden_features=32,
        bilinear=True,
    )

    # Define the input and forward.
    x = torch.randn(1, 3, 256, 256)
    out = model(x)

    print(out)
