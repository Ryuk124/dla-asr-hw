import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram


class LogMelSpectrogram(nn.Module):
    def __init__(self, sample_rate: int = 16000, hop_length: int = 160, normalize: bool = False, n_mels = 80) -> None:
        super().__init__()
        self.mel = MelSpectrogram(sample_rate=sample_rate, hop_length=hop_length, n_mels=n_mels)
        self.normalize = normalize

    def forward(self, audio):
        spectrogram = self.mel(audio)
        log_spectrogram = torch.log(spectrogram + 1e-9)

        if self.normalize:
            mean = log_spectrogram.mean(dim=-1, keepdim=True)
            std = log_spectrogram.std(dim=-1, keepdim=True, correction=0)
            log_spectrogram = (log_spectrogram - mean) / std.clamp_min(1e-5)

        return log_spectrogram 
