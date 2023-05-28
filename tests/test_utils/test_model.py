# encoding: utf-8
import pytest
import torch
import torch.nn as nn
from ahcore.utils.model import ExtractFeaturesHook


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


@pytest.fixture
def dummy_model():
    return DummyModel()


@pytest.fixture
def dummy_input():
    return torch.randn(1, 3, 224, 224)


def test_model_hook(dummy_model, dummy_input):
    with ExtractFeaturesHook(dummy_model, ["conv1", "conv2"]) as hook:
        output = dummy_model(dummy_input)
        conv1_features = hook.features["conv1"]
        conv2_features = hook.features["conv2"]

    # assert that the hook output matches the actual output
    assert torch.allclose(conv1_features, dummy_model.conv1(dummy_input))
    assert torch.allclose(conv2_features, output)
