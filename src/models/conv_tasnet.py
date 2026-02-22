"""Conv-TasNet model for speech enhancement."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1dBlock(nn.Module):
    """1D Convolutional block with normalization and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        batch_norm: bool = True,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        
        self.batch_norm = nn.BatchNorm1d(out_channels) if batch_norm else None
        
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = None
            
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        
        if self.batch_norm is not None:
            x = self.batch_norm(x)
            
        if self.activation is not None:
            x = self.activation(x)
            
        if self.dropout is not None:
            x = self.dropout(x)
            
        return x


class Encoder(nn.Module):
    """Encoder for Conv-TasNet."""
    
    def __init__(
        self,
        encoder_dim: int = 256,
        kernel_size: int = 16,
        stride: int = 8,
    ):
        super().__init__()
        
        self.encoder_dim = encoder_dim
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=encoder_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=False,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input waveform [B, 1, T]
            
        Returns:
            Encoded features [B, encoder_dim, T']
        """
        return self.conv(x)


class Decoder(nn.Module):
    """Decoder for Conv-TasNet."""
    
    def __init__(
        self,
        decoder_dim: int = 256,
        kernel_size: int = 16,
        stride: int = 8,
    ):
        super().__init__()
        
        self.decoder_dim = decoder_dim
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels=decoder_dim,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=False,
        )
    
    def forward(self, x: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            x: Encoded features [B, decoder_dim, T']
            length: Target output length
            
        Returns:
            Reconstructed waveform [B, 1, T]
        """
        output = self.conv_transpose(x)
        
        if length is not None:
            output = output[..., :length]
            
        return output


class Separator(nn.Module):
    """Separator network for Conv-TasNet."""
    
    def __init__(
        self,
        encoder_dim: int = 256,
        n_sources: int = 2,
        n_layers: int = 8,
        n_filters: int = 512,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        batch_norm: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.encoder_dim = encoder_dim
        self.n_sources = n_sources
        self.n_layers = n_layers
        
        # Input projection
        self.input_proj = Conv1dBlock(
            in_channels=encoder_dim,
            out_channels=n_filters,
            kernel_size=1,
            batch_norm=batch_norm,
            activation="relu",
            dropout=dropout,
        )
        
        # Separator layers
        self.separator_layers = nn.ModuleList([
            Conv1dBlock(
                in_channels=n_filters,
                out_channels=n_filters,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                batch_norm=batch_norm,
                activation="relu",
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Conv1d(
            in_channels=n_filters,
            out_channels=encoder_dim * n_sources,
            kernel_size=1,
            bias=False,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Encoded features [B, encoder_dim, T']
            
        Returns:
            Separated features [B, n_sources, encoder_dim, T']
        """
        # Input projection
        x = self.input_proj(x)
        
        # Separator layers
        for layer in self.separator_layers:
            x = layer(x)
        
        # Output projection
        x = self.output_proj(x)
        
        # Reshape to [B, n_sources, encoder_dim, T']
        B, _, T = x.shape
        x = x.view(B, self.n_sources, self.encoder_dim, T)
        
        return x


class ConvTasNet(nn.Module):
    """Conv-TasNet model for speech enhancement."""
    
    def __init__(
        self,
        encoder_dim: int = 256,
        decoder_dim: int = 256,
        encoder_kernel_size: int = 16,
        decoder_kernel_size: int = 16,
        encoder_stride: int = 8,
        decoder_stride: int = 8,
        n_sources: int = 2,
        n_layers: int = 8,
        n_filters: int = 512,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        batch_norm: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.n_sources = n_sources
        
        # Encoder
        self.encoder = Encoder(
            encoder_dim=encoder_dim,
            kernel_size=encoder_kernel_size,
            stride=encoder_stride,
        )
        
        # Separator
        self.separator = Separator(
            encoder_dim=encoder_dim,
            n_sources=n_sources,
            n_layers=n_layers,
            n_filters=n_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            batch_norm=batch_norm,
            dropout=dropout,
        )
        
        # Decoder
        self.decoder = Decoder(
            decoder_dim=decoder_dim,
            kernel_size=decoder_kernel_size,
            stride=decoder_stride,
        )
    
    def forward(
        self,
        mixture: torch.Tensor,
        target_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mixture: Input mixture waveform [B, 1, T]
            target_length: Target output length
            
        Returns:
            Tuple of (enhanced_speech, separated_sources)
        """
        # Encode
        encoded = self.encoder(mixture)
        
        # Separate
        separated = self.separator(encoded)
        
        # Decode each source
        sources = []
        for i in range(self.n_sources):
            source_encoded = separated[:, i]
            source_decoded = self.decoder(source_encoded, target_length)
            sources.append(source_decoded)
        
        # First source is enhanced speech, rest are separated sources
        enhanced_speech = sources[0]
        separated_sources = torch.stack(sources[1:], dim=1) if len(sources) > 1 else None
        
        return enhanced_speech, separated_sources
    
    def get_model_size(self) -> int:
        """Get model size in parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_model_size_mb(self) -> float:
        """Get model size in MB."""
        return self.get_model_size() * 4 / (1024 * 1024)  # Assuming float32
