"""castelens package. Pure-math modules (metrics, datasets, directions self-test)
run without torch; the wrapper is imported lazily so a torch-less machine can still
run metrics.py and datasets.py standalone."""

from .metrics import ches, dcc, dcc_ci, maf, maf_ci, m1, m4, bootstrap_ci
from .directions import build_direction, cosine, orthogonalize, geometry_matched_random


def __getattr__(name):
    if name == "ModelWrapper":
        from .wrapper import ModelWrapper
        return ModelWrapper
    raise AttributeError(name)
