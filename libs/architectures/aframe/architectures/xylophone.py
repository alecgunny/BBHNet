from typing import Optional

import torch

from aframe.architectures.resnet import (
    Bottleneck,
    BottleneckResNet,
    get_norm_layer,
)


class Xylophone(BottleneckResNet):
    def __init__(self, norm_groups: Optional[int] = None):
        torch.nn.Module.__init__(self)
        self._norm_layer = get_norm_layer(norm_groups)

        num_features = 0
        self.octave0, features = self.make_octave(
            32, 8, "dilation", layers=[2, 2], kernel_size=7
        )
        num_features += features

        self.octave1, features = self.make_octave(
            16, 4, "dilation", layers=[3, 3], kernel_size=7
        )
        num_features += features

        self.octave2, features = self.make_octave(
            8, 2, "dilation", layers=[4, 4], kernel_size=7
        )
        num_features += features

        self.octave3, features = self.make_octave(
            8, 3, "stride", layers=[4, 4], kernel_size=7
        )
        num_features += features

        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(num_features, 1)
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, (torch.nn.BatchNorm1d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

            if isinstance(m, Bottleneck):
                torch.nn.init.constant_(m.bn2.weight, 0)

    def make_octave(
        self,
        base_channels: int,
        stride: int,
        stride_type: str,
        layers: list[int],
        kernel_size: int,
    ):
        self.inplanes = 2
        self.dilation = 1
        self.groups = 1
        self.base_width = base_channels
        residual_layers = [
            self._make_layer(base_channels, layers[0], kernel_size)
        ]
        for i, num_blocks in enumerate(layers[1:]):
            block_size = base_channels * 2 ** (i + 1)
            layer = self._make_layer(
                block_size,
                num_blocks,
                kernel_size,
                stride=stride,
                stride_type=stride_type,
            )
            residual_layers.append(layer)

        return (
            torch.nn.Sequential(*residual_layers),
            block_size * self.block.expansion,
        )

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        size = x.size(-1)
        octave_size = int(2 * size // 5)
        octave_step = int(octave_size // 2)

        outputs = []
        for i in range(4):
            start = i * octave_step
            octave = x[:, :, start : start + octave_size]
            octave = getattr(self, f"octave{i}")(octave)
            octave = self.avgpool(octave)
            outputs.append(octave)
        x = torch.cat(outputs, axis=1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
