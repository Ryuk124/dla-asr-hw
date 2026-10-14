from torch import Tensor
import torch.nn as nn
import torchaudio.transforms as T


# These two don't make sense, added for compatibility with the rest of the code
class FrequencyMasking(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = T.FrequencyMasking(*args, **kwargs)

    def forward(self, data: Tensor):
        return self._aug(data)


class TimeMasking(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = T.TimeMasking(*args, **kwargs)

    def forward(self, data: Tensor):
        return self._aug(data)
