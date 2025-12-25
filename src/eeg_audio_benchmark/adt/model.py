"""ADT-style Transformer model for EEG→audio envelope decoding."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from .config import ADTModelConfig


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention over EEG channels."""

    def __init__(self, n_channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, n_channels // reduction)
        self.fc1 = nn.Linear(n_channels, hidden)
        self.fc2 = nn.Linear(hidden, n_channels)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, T, C]
        weights = x.mean(dim=1)  # [batch, C]
        weights = self.fc2(self.activation(self.fc1(weights)))
        weights = self.sigmoid(weights).unsqueeze(1)
        return x * weights


class SinusoidalPositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding for Transformer inputs."""

    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1))  # [max_len, 1, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, batch, d_model]
        T = x.size(0)
        if T > self.pe.size(0):
            raise ValueError(f"Sequence length {T} exceeds maximum supported {self.pe.size(0)}")
        return x + self.pe[:T]


class EEGToEnvelopeADT(nn.Module):
    """Transformer baseline mapping EEG time series to acoustic envelope features."""

    def __init__(self, n_channels: int, out_dim: int, config: ADTModelConfig):
        super().__init__()
        self.config = config
        kernel_size = 25
        padding = kernel_size // 2
        self.channel_attention: Optional[ChannelAttention] = None
        if config.use_channel_attention:
            self.channel_attention = ChannelAttention(n_channels=n_channels)

        self.conv = nn.Conv1d(
            in_channels=n_channels,
            out_channels=config.d_model,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv_activation = nn.GELU()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.d_model),
        )
        self.positional_encoding = SinusoidalPositionalEncoding(d_model=config.d_model)
        self.output_proj = nn.Linear(config.d_model, out_dim)
        self.dropout = nn.Dropout(config.dropout)

    def _causal_mask(self, T: int, device: torch.device) -> Optional[torch.Tensor]:
        if not self.config.causal:
            return None
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Tensor of shape [batch, T, C] containing EEG samples.

        Returns
        -------
        torch.Tensor
            Predicted acoustic envelope features with shape [batch, T, out_dim].
        """

        if self.channel_attention is not None:
            x = self.channel_attention(x)
        # Conv1d expects [batch, C, T]
        x = x.transpose(1, 2)
        x = self.conv_activation(self.conv(x))
        x = x.transpose(1, 2)  # [batch, T, d_model]

        x = self.dropout(x)
        x = x.transpose(0, 1)  # [T, batch, d_model]
        x = self.positional_encoding(x)
        mask = self._causal_mask(x.size(0), device=x.device)
        x = self.transformer(x, mask=mask)
        x = x.transpose(0, 1)  # [batch, T, d_model]
        x = self.output_proj(x)
        return x


__all__ = ["EEGToEnvelopeADT"]
