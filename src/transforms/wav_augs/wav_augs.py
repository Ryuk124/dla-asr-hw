import torch_audiomentations
from torch import Tensor, nn
from torchaudio.transforms import SpeedPerturbation


class Gain(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = torch_audiomentations.Gain(*args, **kwargs)

    def forward(self, data: Tensor):
        x = data.unsqueeze(0)
        return self._aug(x).squeeze(0)


class SpeedPerturbationAug(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = SpeedPerturbation(*args, **kwargs)

    def forward(self, data: Tensor):
        x, _ = self._aug(data)
        return x


class AddColoredNoise(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = torch_audiomentations.AddColoredNoise(*args, **kwargs)

    def forward(self, data: Tensor):
        x = data.unsqueeze(0)
        return self._aug(x).squeeze(0)


class AddBackgroundNoise(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = torch_audiomentations.AddBackgroundNoise(*args, **kwargs)

    def forward(self, data: Tensor):
        x = data.unsqueeze(0)
        return self._aug(x).squeeze(0)


class ApplyImpulseResponse(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = torch_audiomentations.ApplyImpulseResponse(*args, **kwargs)

    def forward(self, data: Tensor):
        x = data.unsqueeze(0)
        return self._aug(x).squeeze(0)
