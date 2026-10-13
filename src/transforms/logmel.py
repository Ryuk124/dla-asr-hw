import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram


class LogMelSpectrogram(nn.Module):
    def __init__(self, sample_rate: int = 16000) -> None:
        super().__init__()
        self.mel = MelSpectrogram(sample_rate=sample_rate)

    def forward(self, audio):
        spectrogram = self.mel(audio)
        return torch.log(spectrogram + 1e-9)
