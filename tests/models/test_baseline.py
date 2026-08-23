import torch
from dsfnet.models.baseline import SpatialOnlyClassifier


def test_baseline_classifier_output_shape():
    model = SpatialOnlyClassifier(pretrained=False, num_classes=2)
    x = torch.rand(3, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (3, 2)
