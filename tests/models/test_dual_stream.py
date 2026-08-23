import torch
from dsfnet.models.dual_stream import DualStreamNet
from dsfnet.models.spatial_stream import SpatialStream
from dsfnet.models.frequency_stream import FrequencyStream


def test_dual_stream_output_shape():
    model = DualStreamNet(pretrained_spatial=False, num_classes=2)
    x = torch.rand(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, 2)


def test_fused_dim_matches_report_768():
    assert SpatialStream.EMBED_DIM + FrequencyStream.EMBED_DIM == 768


def test_gradients_flow_into_both_streams():
    model = DualStreamNet(pretrained_spatial=False)
    x = torch.rand(1, 3, 224, 224)
    logits = model(x)
    logits.sum().backward()
    assert model.spatial.features[0].weight.grad is not None
    assert model.frequency.conv1.weight.grad is not None
