from model_correction.base_correction_method import Vanilla
from model_correction.clarc import AClarc, Clarc, PClarc, ReactivePClarc
from model_correction.rrc import RRClarc
from model_correction.rrr import RRR_CE, RRR_CE_L1, RRR_ExpAll, RRR_ExpLogSum, RRR_ExpMax, RRR_ExpMax_L1, RRR_ExpSoftmax, RRR_ExpTarget


def get_correction_method(method_name):
    CORRECTION_METHODS = {
        'Vanilla': Vanilla,
        'Clarc': Clarc,
        'AClarc': AClarc,
        'PClarc': PClarc,
        'ReactivePClarc': ReactivePClarc,
        'RRClarc': RRClarc,
        'RRR_CE': RRR_CE,
        'RRR_CE_L1': RRR_CE_L1,
        'RRR_ExpMax': RRR_ExpMax,
        'RRR_ExpMax_L1': RRR_ExpMax_L1,
        'RRR_ExpTarget': RRR_ExpTarget,
        'RRR_ExpAll': RRR_ExpAll,
        'RRR_ExpSoftmax': RRR_ExpSoftmax,
        'RRR_ExpLogSum': RRR_ExpLogSum,

    }

    assert method_name in CORRECTION_METHODS.keys(), f"Correction method '{method_name}' unknown," \
                                                     f" choose one of {list(CORRECTION_METHODS.keys())}"
    return CORRECTION_METHODS[method_name]
