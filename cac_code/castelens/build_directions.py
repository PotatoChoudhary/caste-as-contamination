"""
castelens/build_directions.py — construct V1..V7, the random null (V6), and the
geometry-matched randoms, all at the chosen layer. Every experiment imports this.

Each direction = difference of means on its dataset's pos/neg arms, EXTRACT fold only.
Returns a dict you pass around. Pure plumbing over the wrapper + frozen data.
"""
from .directions import build_direction, geometry_matched_random
from . import config
import torch


def build_all(wrapper, data, layer):
    """
    data: dict of loaded frozen datasets, each with 'pos'/'neg' prompt lists, e.g.
        data['D1'] = {'pos':[...marked...], 'neg':[...neutral...]}   -> V1
        data['D2'] -> V2 (labels), D4->V3 (disgust), D5->V4 (anger),
        data['D6'] -> V5 (valence), D10->V7 (status)
    Returns {'V1':t, 'V2':t, 'V3':t, 'V4':t, 'V5':t, 'V7':t, 'V6':[...], 'rand_matched':[...]}
    """
    spec = {"V1": "D1", "V2": "D2", "V3": "D4", "V4": "D5", "V5": "D6", "V7": "D10"}
    V = {k: build_direction(wrapper, data[ds]["pos"], data[ds]["neg"], layer)
         for k, ds in spec.items()}

    # V6: random null directions (unit, in model space)
    g = torch.Generator().manual_seed(0)
    V["V6"] = [torch.randn(wrapper.d_model, generator=g) for _ in range(config.N_RANDOM_NULL)]
    V["V6"] = [r / r.norm() for r in V["V6"]]

    # geometry-matched randoms (overlap V1 exactly as much as V3 does)
    V["rand_matched"] = geometry_matched_random(V["V1"], V["V3"], n=config.N_MATCHED_RANDOM)
    return V
