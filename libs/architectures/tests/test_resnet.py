import pytest
import torch

from bbhnet.architectures.resnet import (
    BasicBlock,
    Bottleneck,
    BottleneckResNet,
    ResNet,
    conv1,
)


@pytest.fixture(params=[3, 7, 8, 11])
def kernel_size(request):
    return request.param


@pytest.fixture(params=[1024, 4096])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[1, 2])
def stride(request):
    return request.param


@pytest.fixture(params=[2, 4])
def inplanes(request):
    return request.param


@pytest.fixture(params=[BasicBlock, Bottleneck])
def block(request):
    return request.param


def test_blocks(block, kernel_size, stride, sample_rate, inplanes):
    # TODO: test dilation for bottleneck
    planes = 4

    if stride > 1 or inplanes != planes * block.expansion:
        downsample = conv1(inplanes, planes * block.expansion, stride)
    else:
        downsample = None

    if kernel_size % 2 == 0:
        with pytest.raises(ValueError):
            block = block(
                inplanes, planes, kernel_size, stride, downsample=downsample
            )
        return

    block = block(inplanes, planes, kernel_size, stride, downsample=downsample)
    x = torch.randn(8, inplanes, sample_rate)
    y = block(x)

    assert len(y.shape) == 3
    assert y.shape[1] == planes * block.expansion
    assert y.shape[2] == sample_rate // stride


@pytest.fixture(params=[1, 2, 3])
def num_ifos(request):
    return request.param


@pytest.fixture(params=[[2, 2, 2, 2], [2, 4, 4], [3, 4, 6, 3]])
def layers(request):
    return request.param


@pytest.fixture(params=[None, "stride", "dilation"])
def stride_type(request):
    return request.param


@pytest.fixture(params=[BottleneckResNet, ResNet])
def architecture(request):
    return request.param


def test_resnet(
    architecture, kernel_size, layers, num_ifos, sample_rate, stride_type
):
    if kernel_size % 2 == 0:
        with pytest.raises(ValueError):
            nn = ResNet(num_ifos, layers, kernel_size)
        return

    if stride_type is not None:
        stride_type = [stride_type] * (len(layers) - 1)

    if (
        stride_type is not None
        and stride_type[0] == "dilation"
        and architecture == ResNet
    ):
        with pytest.raises(NotImplementedError):
            nn = architecture(
                num_ifos, layers, kernel_size, stride_type=stride_type
            )
        return

    nn = architecture(num_ifos, layers, kernel_size, stride_type=stride_type)
    x = torch.randn(8, num_ifos, sample_rate)
    y = nn(x)
    assert y.shape == (8, 1)

    with pytest.raises(ValueError):
        stride_type = ["stride"] * len(layers)
        nn = architecture(
            num_ifos, layers, kernel_size, stride_type=stride_type
        )
    with pytest.raises(ValueError):
        stride_type = ["strife"] * (len(layers) - 1)
        nn = architecture(
            num_ifos, layers, kernel_size, stride_type=stride_type
        )
