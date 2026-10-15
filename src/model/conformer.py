import torch
from torch import Tensor, nn
from torchtune.modules import RotaryPositionalEmbeddings


# Original Conformer realisation from https://arxiv.org/pdf/2005.08100
# but with rmsnorm instead of batch/layer norm everywhere and rope instead of sinusoidal pos embeds
class ConformerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_tokens: int,
        n_features: int,
        blocks_amount: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        att_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = ConformerEncoder(
            d_model=d_model,
            n_features=n_features,
            blocks_amount=blocks_amount,
            num_heads=num_heads,
            dropout=dropout,
            att_dropout=att_dropout,
        )

        self.decoder = nn.Sequential(
            nn.Linear(d_model, n_tokens), nn.LogSoftmax(dim=-1)
        )

    def forward(self, spectrogram, spectrogram_length, **batch):
        x, lengths = self.encoder(spectrogram, spectrogram_length)
        return {"log_probs": self.decoder(x), "log_probs_length": lengths}


class ConformerEncoder(nn.Module):
    """Encode spectrograms with subsampling and Conformer blocks.

    Args:
        d_model: Encoder feature dimension, divisible by num_heads.
        n_features: Number of input mel bins.
        blocks_amount: Number of Conformer blocks.
        num_heads: Number of attention heads. Defaults to 8.
        dropout: Dropout probability for subsampling, FFNs, and convolution.
            Defaults to 0.1.
        att_dropout: Dropout probability for attention weights and the attention
            module output. Defaults to 0.1.
    """

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
        self.conv = ConvolutionalSubsampling(
            d_model=d_model, n_features=n_features, dropout=dropout
        )
        self.conv_blocks = nn.ModuleList(
            [
                ConformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                    att_dropout=att_dropout,
                )
                for _ in range(blocks_amount)
            ]
        )

    def forward(self, x, spectrogram_length):
        """Encode a batch of padded spectrograms.

        Args:
            x (torch.Tensor): Spectrograms of shape [B, n_features, T].
            spectrogram_length (torch.Tensor): Integer lengths of shape [B],
                measured in input frames before padding and subsampling.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Encoded features of shape
            [B, T_out, d_model] and subsampled lengths of shape [B]. Both
            tensors are on the same device as x.
        """
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
    """Subsample time and frequency with two strided convolutions.

    Args:
        d_model: Convolution channel count and output feature dimension.
        n_features: Number of input mel bins. Must be at least 7.
        dropout: Dropout probability after projection. Defaults to 0.1.
    """

    def __init__(self, d_model: int, n_features: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=d_model, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=d_model, out_channels=d_model, kernel_size=3, stride=2
            ),
            nn.ReLU(),
        )

        out_freq = (n_features - 1) // 2
        out_freq = (out_freq - 1) // 2

        self.ffn = nn.Linear(in_features=d_model * out_freq, out_features=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """Subsample spectrograms and project them to the encoder dimension.

        Args:
            x (torch.Tensor): Spectrograms of shape [B, n_features, T],
                where T is at least 7.

        Returns:
            torch.Tensor: Features of shape [B, T_out, d_model], where
            T_out = ((T - 1) // 2 - 1) // 2.
        """
        x = x.transpose(1, 2).unsqueeze(1)  # [B, F, T] -> [B, 1, T, F]

        x = self.conv(x)

        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3)  # [B, T, C, F]
        x = x.reshape(B, T, C * F)

        x = self.dropout(self.ffn(x))

        return x


class ConformerBlock(nn.Module):
    """Apply two FFNs, self-attention, convolution, and a final RMSNorm.

    Each submodule includes its residual connection; FFNs use a 0.5 scale.

    Args:
        d_model: Feature dimension, divisible by num_heads.
        num_heads: Number of attention heads. Defaults to 8.
        dropout: Dropout probability for FFNs and convolution. Defaults to 0.1.
        att_dropout: Dropout probability for attention weights and the attention
            module output. Defaults to 0.1.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        att_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ffn1 = FeedForwardBlock(d_model=d_model, dropout=dropout)
        self.mhsa = MultiHeadAttentionBlock(
            d_model=d_model, num_heads=num_heads, dropout=att_dropout
        )
        self.conv = ConvolutionalModule(d_model=d_model, dropout=dropout)
        self.ffn2 = FeedForwardBlock(d_model=d_model, dropout=dropout)
        self.norm = nn.RMSNorm(normalized_shape=d_model)

    def forward(self, x, pad_mask):
        """Apply the block while preserving sequence length.

        Args:
            x (torch.Tensor): Features of shape [B, T, d_model].
            pad_mask (torch.Tensor): Boolean mask of shape [B, T], on the same
                device as x. True marks padded positions.

        Returns:
            torch.Tensor: Features of shape [B, T, d_model]. Padded output
            positions are not guaranteed to be zero.
        """
        x = self.ffn1(x)
        x = self.mhsa(x, pad_mask)
        x = self.conv(x, pad_mask)
        x = self.ffn2(x)
        x = self.norm(x)

        return x


class FeedForwardBlock(nn.Module):
    """Apply a pre-norm FFN with a half-step residual connection.

    Args:
        d_model: Input and output feature dimension.
        dropout: Dropout probability after the activation and output projection.
    """

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()

        self.ff_block = nn.Sequential(
            nn.RMSNorm(normalized_shape=d_model),
            nn.Linear(in_features=d_model, out_features=d_model * 4),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=d_model * 4, out_features=d_model),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        """Add half of the FFN output to the input.

        Args:
            x (torch.Tensor): Features of shape [B, T, d_model].

        Returns:
            torch.Tensor: Features of shape [B, T, d_model], including residual.
        """
        residual = x
        x = self.ff_block(x)
        return 1 / 2 * x + residual


class ConvolutionalModule(nn.Module):
    """Apply pointwise and depthwise convolutions with a residual connection.

    Padding is zeroed before the depthwise convolution, not at the output.

    Args:
        d_model: Input and output feature dimension.
        dropout: Dropout probability before residual addition.
    """

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(normalized_shape=d_model)
        self.conv1 = nn.Conv1d(
            in_channels=d_model, out_channels=d_model * 2, kernel_size=1
        )
        self.glu = nn.GLU(dim=1)
        self.conv2 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=32,
            groups=d_model,
            padding="same",
        )
        self.norm2 = nn.RMSNorm(normalized_shape=d_model)
        self.silu = nn.SiLU()
        self.conv3 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        """Mix local temporal context without reading padded features.

        Args:
            x (torch.Tensor): Features of shape [B, T, d_model].
            mask (torch.Tensor): Boolean mask of shape [B, T], on the same
                device as x. True marks padded positions.

        Returns:
            torch.Tensor: Features of shape [B, T, d_model], including residual.
            Padded output positions are not guaranteed to be zero.
        """
        residual = x
        x = self.norm1(x)
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.glu(self.conv1(x))

        # Prevent padded positions from leaking into valid frames through convolution.
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
    """Apply pre-norm self-attention with RoPE and a residual connection.

    Rotary positional embeddings are applied to queries and keys within each
    head. Attention is bidirectional; padded keys are excluded when a mask
    is provided.

    Args:
        d_model: Input and output feature dimension. Must be divisible by
            num_heads, with an even dimension per head.
        num_heads: Number of attention heads. Defaults to 8.
        max_pos_enc_len: Maximum supported sequence length after subsampling.
            Defaults to 4096.
        dropout: Dropout probability for attention weights and the output
            projection. Defaults to 0.1.
        bias: Whether to include biases in the QKV and output projections.
            Defaults to False.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        max_pos_enc_len: int = 4096,
        dropout: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_pos_enc_len = max_pos_enc_len
        assert self.head_dim % 2 == 0

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.norm = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout_p = dropout

        self.rope = RotaryPositionalEmbeddings(
            dim=self.head_dim,
            max_seq_len=self.max_pos_enc_len,
            base=10000,
        )

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Tensor | None = None,
    ):
        """Apply rotary self-attention while preserving the input shape.

        Args:
            x (Tensor): Input features of shape [B, T, d_model].
            key_padding_mask (Tensor | None): Boolean mask of shape [B, T],
                on the same device as x. True marks padded keys that attention
                must ignore. Defaults to None.

        Returns:
            Tensor: Features of shape [B, T, d_model], including the residual
            connection. Padded query positions are not guaranteed to be zero.
        """
        x_norm = self.norm(x)
        attn_mask = (
            ~key_padding_mask[:, None, None, :]
            if key_padding_mask is not None
            else None
        )

        q, k, v = self.qkv_proj(x_norm).chunk(3, dim=-1)
        B, T, D = q.shape
        q = q.reshape(B, T, self.num_heads, self.head_dim) # [B, T, H, h_dim]
        k = k.reshape(B, T, self.num_heads, self.head_dim)
        v = v.reshape(B, T, self.num_heads, self.head_dim)

        q = self.rope(q)
        k = self.rope(k)

        q = q.transpose(1, 2) # [B, H, T, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_p if self.training else 0.0
        )

        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.out_proj(out)
        out = self.dropout(out)

        return out + x

