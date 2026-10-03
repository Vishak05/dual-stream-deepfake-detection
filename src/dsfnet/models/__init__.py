from functools import partial

from .baseline import SpatialOnlyClassifier
from .dual_stream import DualStreamNet

# Name -> constructor, shared by every training/evaluation script.
MODELS = {
    "spatial_only": SpatialOnlyClassifier,
    "dual_stream": DualStreamNet,
    "dual_stream_v2": partial(DualStreamNet, frequency_mode="mean_spectrum"),
}
