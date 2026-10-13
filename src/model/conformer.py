from torch import nn
import torch
from torch.nn import Sequential


# Original Conformer realisation from https://arxiv.org/pdf/2005.08100
# but without pos embeds and with rmsnorm instead of batch/layer norm everywhere
class ConformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_features: int,
        blocks_amount: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        att_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv = ConvolutionalSubsampling(d_model, n_features, dropout)
        self.conv_blocks = nn.ModuleList(
            [
                ConformerBlock(d_model, num_heads, dropout, att_dropout)
                for _ in range(blocks_amount)
            ]
        )

    def forward(self, x, spectrogram_length):
        x = self.conv(x)

        lengths = (spectrogram_length - 1) // 2
        lengths = (lengths - 1) // 2
        lengths = lengths.to(x.device)

        positions = torch.arange(x.size(1), device=x.device)  # [T_new]
        pad_mask = positions.unsqueeze(0) >= lengths.unsqueeze(1)  # [B, T_new]

        for block in self.conv_blocks:
            x = block(x, pad_mask)

        return x, lengths


class ConvolutionalSubsampling(nn.Module):
    def __init__(self, d_model: int, n_features: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        out_freq = (n_features - 1) // 2
        out_freq = (out_freq - 1) // 2

        self.ffn = nn.Linear(d_model * out_freq, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2).unsqueeze(1)  # [B, F, T] from collate_fn to [B, 1, T, F]

        x = self.conv(x)  # [B, C, T, F]

        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3)  # [B, T, C, F]
        x = x.reshape(B, T, C * F)

        x = self.dropout(self.ffn(x))  # [B, T, D]

        return x


class ConformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        att_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ffn1 = FeedForwardBlock(d_model, dropout)
        self.mhsa = MultiHeadAttentionBlock(
            d_model, num_heads=num_heads, att_dropout=att_dropout
        )
        self.conv = ConvolutionalModule(d_model, dropout)
        self.ffn2 = FeedForwardBlock(d_model, dropout)
        self.norm = nn.RMSNorm(d_model)

    def forward(self, x, pad_mask):
        x = self.ffn1(x)
        x = self.mhsa(x, pad_mask)
        x = self.conv(x, pad_mask)
        x = self.ffn2(x)
        x = self.norm(x)

        return x


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()

        self.ff_block = Sequential(
            nn.RMSNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        residual = x
        x = self.ff_block(x)
        return 1 / 2 * x + residual


class ConvolutionalModule(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.conv2 = nn.Conv1d(
            d_model, d_model, kernel_size=32, groups=d_model, padding="same"
        )
        self.norm2 = nn.RMSNorm(d_model)
        self.silu = nn.SiLU()
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        residual = x
        x = self.norm1(x)
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.glu(self.conv1(x))

        x = x.masked_fill(mask.unsqueeze(1), 0.0)
        x = self.conv2(x)
        x = x.transpose(1, 2)  # [B, T, D]
        x = self.silu(self.norm2(x))

        x = x.transpose(1, 2)  # [B, D, T]
        x = self.conv3(x)

        x = x.transpose(1, 2)  # [B, T, D]
        x = self.dropout(x)

        return x + residual


class MultiHeadAttentionBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int = 8, att_dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.norm = nn.RMSNorm(d_model)
        self.att = nn.MultiheadAttention(
            d_model, num_heads=num_heads, batch_first=True, dropout=att_dropout
        )
        self.dropout = nn.Dropout(att_dropout)

    def forward(self, x, key_padding_mask):
        residual = x
        x = self.norm(x)
        x, _ = self.att(x, x, x, need_weights=False, key_padding_mask=key_padding_mask)
        x = self.dropout(x)
        return x + residual
