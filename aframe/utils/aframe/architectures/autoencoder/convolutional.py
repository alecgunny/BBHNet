import functools
from collections.abc import Callable, Sequence
from typing import Optional

import torch

from aframe.architectures.autoencoder.base import Autoencoder
from aframe.architectures.autoencoder.skip_connection import SkipConnection
from aframe.architectures.autoencoder.utils import match_size

Module = Callable[[...], torch.nn.Module]


class ConvBlock(Autoencoder):
    def __init__(
        self,
        in_channels: int,
        encode_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        activation: torch.nn.Module = torch.nn.ReLU(),
        norm: Module = torch.nn.BatchNorm1d,
        decode_channels: Optional[int] = None,
        output_activation: Optional[torch.nn.Module] = None,
        skip_connection: Optional[SkipConnection] = None
    ) -> None:
        super().__init__(skip_connection=None)

        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) // 2)
        self.stride = stride

        out_channels = encode_channels * groups
        self.encode_layer = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=False,
            groups=groups
        )

        decode_channels = decode_channels or in_channels
        in_channels = encode_channels * groups
        if skip_connection is not None:
            in_channels = skip_connection.get_out_channels(in_channels)
        self.decode_layer = torch.nn.ConvTranspose1d(
            in_channels,
            decode_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=False,
            groups=groups
        )

        self.activation = activation
        if output_activation is not None:
            self.output_activation = output_activation
        else:
            self.output_activation = activation

        self.norm1 = norm(out_channels)
        self.norm2 = norm(decode_channels)

    def encode(self, X):
        X = self.encode_layer(X)
        X = self.norm1(X)
        return self.activation(X)

    def decode(self, X):
        X = self.decode_layer(X)
        X = self.norm2(X)
        return self.output_activation(X)


class ConvolutionalAutoencoder(Autoencoder):
    def __init__(
        self,
        num_ifos: int,
        encode_channels: Sequence[int],
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        activation: torch.nn.Module = torch.nn.ReLU(),
        output_activation: Optional[torch.nn.Module] = None,
        norm: Module = torch.nn.BatchNorm1d,
        decode_channels: Optional[int] = None,
        skip_connection: Optional[SkipConnection] = None
    ) -> None:
        # TODO: how to do this dynamically? Maybe the base
        # architecture looks for overlapping arguments between
        # this and the skip connection class and then provides them?
        # if skip_connection is not None:
        #     skip_connection = skip_connection(groups)
        super().__init__(skip_connection=skip_connection)

        output_activation = output_activation or activation
        in_channels = num_ifos
        for i, channels in enumerate(encode_channels):
            j = len(encode_channels) - 1 - i
            block = ConvBlock(
                in_channels,
                channels,
                kernel_size,
                stride,
                groups,
                activation=activation,
                norm=norm,
                decode_channels=decode_channels if not i else in_channels,
                skip_connection=skip_connection if j else None,
                output_activation=None if i else output_activation
            )
            self.blocks.append(block)
            in_channels = channels * groups

    def decode(self, *X, states=None, input_size: Optional[int] = None):
        X = super().decode(*X, states=states)
        if input_size is not None:
            return match_size(X, input_size)
        return X

    def forward(self, X):
        input_size = X.size(-1)
        X = super().forward(X)
        return match_size(X, input_size)
