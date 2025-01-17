import numpy as np
from corelay.processor.distance import SciPyPDist
from corelay.processor.base import Processor
from corelay.base import Param
from corelay.processor.flow import Sequential
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity

class Flatten(Processor):
    def function(self, data):
        return data.reshape(data.shape[0], np.prod(data.shape[1:]))


class SumChannel(Processor):
    def function(self, data):
        return data.sum(1)


class Absolute(Processor):
    def function(self, data):
        return np.absolute(data)


class Normalize(Processor):
    axes = Param(tuple, (1,))

    def function(self, data):
        print(data.shape)
        data = data / data.sum(self.axes, keepdims=True)
        return data
    
class Normalize2d(Processor):
    axes = Param(tuple, (1, 2))

    def function(self, data):
        print(data.shape)
        data = data / data.sum(self.axes, keepdims=True)
        return data


def csints(string):
    return tuple(int(elem) for elem in string.split(','))


class Histogram(Processor):
    bins = Param(int, 256)

    def function(self, data):
        hists = np.stack([
            np.stack([
                np.histogram(
                    arr.reshape(arr.shape[0], np.prod(arr.shape[1:3])),
                    bins=self.bins,
                    density=True
                ) for arr in channel
            ]) for channel in data.transpose(3, 0, 1, 2)])
        return hists


class PCC(Processor):
    def function(self, data):
        return squareform(pdist(data, metric=lambda x, y: pearsonr(x, y)[0]))


class SSIM(Processor):
    def function(self, data):
        N, H, W = data.shape
        return squareform(pdist(
            data.reshape(N, H * W),
            metric=lambda x, y: structural_similarity(x.reshape(H, W), y.reshape(H, W), multichannel=False)
        ))


VARIANTS = {
    'absspectral': {
        'preprocessing': Sequential([
            Absolute(),
            SumChannel(),
            Normalize(),
            Flatten()
        ]),
        'distance': SciPyPDist(metric='euclidean'),
    },
    'spectral': {
        'preprocessing': Sequential([
            # SumChannel(),
            Normalize(),
            Flatten()
        ]),
        'distance': SciPyPDist(metric='euclidean'),
    },
    'spectral2d': {
        'preprocessing': Sequential([
            # SumChannel(),
            Normalize2d(),
            Flatten()
        ]),
        'distance': SciPyPDist(metric='euclidean'),
    },
    'fullspectral': {
        'preprocessing': Sequential([
            Normalize(axes=(1, 2, 3)),
            Flatten()
        ]),
        'distance': SciPyPDist(metric='euclidean'),
    },
    'histogram': {
        'preprocessing': Sequential([
            Normalize(axes=(1, 2, 3)),
            Histogram(),
            Flatten()
        ]),
        'distance': SciPyPDist(metric='euclidean'),
    },
    'ssim': {
        'preprocessing': Sequential([
            SumChannel(),
            Normalize(),
        ]),
        'distance': SSIM(),
    },
    'pcc': {
        'preprocessing': Sequential([
            SumChannel(),
            Normalize(),
            Flatten()
        ]),
        'distance': PCC(),
    }
}