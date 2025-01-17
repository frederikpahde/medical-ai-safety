import torch
from zennit.composites import SpecialFirstLayerMapComposite, LAYER_MAP_BASE, LayerMapComposite
from zennit.rules import Gamma, Flat, Epsilon
from zennit.types import Convolution, Linear

from utils.lrp_canonizers import NewAttention


class EpsilonGammaFlat(SpecialFirstLayerMapComposite):
    '''An explicit composite using the flat rule for any linear first layer, the zplus rule for all other convolutional
    layers and the epsilon rule for all other fully connected layers.
    '''

    def __init__(self, canonizers=None):
        layer_map = LAYER_MAP_BASE + [
            (Convolution, Gamma(0.25)),
            (torch.nn.Linear, Epsilon()),
        ]
        first_map = [
            (Linear, Flat())
        ]
        super().__init__(layer_map, first_map, canonizers=canonizers)


class EpsilonPlusFlat(SpecialFirstLayerMapComposite):
    '''An explicit composite using the flat rule for any linear first layer, the zplus rule for all other convolutional
    layers and the epsilon rule for all other fully connected layers.
    '''

    def __init__(self, canonizers=None):
        layer_map = \
            [
                (torch.nn.LayerNorm, Epsilon(0.1)),
                (NewAttention, Epsilon(0.1)),
                (torch.nn.MultiheadAttention, Epsilon(0.1)),
            ] + \
            LAYER_MAP_BASE + [
                (Convolution, Gamma(0.25)),
                (torch.nn.Linear, Epsilon(0.1)),
                # (nn.LayerNorm, Epsilon()),
            ]
        first_map = [
            # (Linear, Flat())
        ]
        super().__init__(layer_map, first_map, canonizers=canonizers)

class EpsilonComposite(SpecialFirstLayerMapComposite):
    '''An explicit composite using the epsilon rule for any layer.
    '''
    def __init__(self, canonizers=None):
        layer_map = LAYER_MAP_BASE + [
            (Convolution, Epsilon()),
            (torch.nn.Linear, Epsilon()),
            # (BatchNorm, Pass()),
        ]
        first_map = [
            (Linear, Epsilon())
        ]
        super().__init__(layer_map, first_map, canonizers=canonizers)


class GradientComposite(LayerMapComposite):
    '''An explicit composite to compute the gradient.
    '''

    def __init__(self, canonizers=None):
        layer_map = [
        ]
        super().__init__(layer_map, canonizers=canonizers)