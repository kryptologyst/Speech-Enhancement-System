"""Spectral subtraction and Wiener filtering for speech enhancement."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SpectralSubtraction(nn.Module):
    """Spectral subtraction for speech enhancement."""
    
    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: Optional[int] = None,
        window: str = "hann",
        alpha: float = 2.0,
        beta: float = 0.01,
        gamma: float = 0.1,
    ):
        super().__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.window = window
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Initialize window
        if window == "hann":
            self.register_buffer("window", torch.hann_window(self.win_length))
        elif window == "hamming":
            self.register_buffer("window", torch.hamming_window(self.win_length))
        else:
            self.register_buffer("window", torch.ones(self.win_length))
    
    def forward(
        self,
        noisy_signal: torch.Tensor,
        noise_estimate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            noisy_signal: Noisy speech signal [B, T]
            noise_estimate: Noise estimate [B, T] or None for auto-estimation
            
        Returns:
            Enhanced speech signal [B, T]
        """
        # Compute STFT
        noisy_stft = torch.stft(
            noisy_signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        
        # Estimate noise if not provided
        if noise_estimate is None:
            noise_estimate = self._estimate_noise(noisy_stft)
        else:
            noise_stft = torch.stft(
                noise_estimate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                return_complex=True,
            )
        
        # Compute magnitude spectra
        noisy_mag = torch.abs(noisy_stft)
        noise_mag = torch.abs(noise_stft) if noise_estimate is not None else noise_estimate
        
        # Spectral subtraction
        enhanced_mag = self._spectral_subtraction(noisy_mag, noise_mag)
        
        # Reconstruct phase
        phase = torch.angle(noisy_stft)
        enhanced_stft = enhanced_mag * torch.exp(1j * phase)
        
        # ISTFT
        enhanced_signal = torch.istft(
            enhanced_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            length=noisy_signal.shape[-1],
        )
        
        return enhanced_signal
    
    def _estimate_noise(self, noisy_stft: torch.Tensor) -> torch.Tensor:
        """Estimate noise from noisy signal using first few frames."""
        # Use first 10% of frames as noise estimate
        n_frames = noisy_stft.shape[-1]
        n_noise_frames = max(1, n_frames // 10)
        
        noise_stft = noisy_stft[..., :n_noise_frames]
        noise_mag = torch.abs(noise_stft).mean(dim=-1, keepdim=True)
        
        return noise_mag
    
    def _spectral_subtraction(
        self,
        noisy_mag: torch.Tensor,
        noise_mag: torch.Tensor,
    ) -> torch.Tensor:
        """Apply spectral subtraction."""
        # Oversubtraction factor
        alpha = self.alpha
        
        # Spectral floor
        beta = self.beta
        
        # Subtraction
        enhanced_mag = noisy_mag - alpha * noise_mag
        
        # Apply spectral floor
        spectral_floor = beta * noisy_mag
        enhanced_mag = torch.maximum(enhanced_mag, spectral_floor)
        
        return enhanced_mag


class WienerFilter(nn.Module):
    """Wiener filter for speech enhancement."""
    
    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: Optional[int] = None,
        window: str = "hann",
        alpha: float = 0.98,
        beta: float = 0.01,
    ):
        super().__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.window = window
        self.alpha = alpha
        self.beta = beta
        
        # Initialize window
        if window == "hann":
            self.register_buffer("window", torch.hann_window(self.win_length))
        elif window == "hamming":
            self.register_buffer("window", torch.hamming_window(self.win_length))
        else:
            self.register_buffer("window", torch.ones(self.win_length))
    
    def forward(
        self,
        noisy_signal: torch.Tensor,
        noise_estimate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            noisy_signal: Noisy speech signal [B, T]
            noise_estimate: Noise estimate [B, T] or None for auto-estimation
            
        Returns:
            Enhanced speech signal [B, T]
        """
        # Compute STFT
        noisy_stft = torch.stft(
            noisy_signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        
        # Estimate noise if not provided
        if noise_estimate is None:
            noise_estimate = self._estimate_noise(noisy_stft)
        else:
            noise_stft = torch.stft(
                noise_estimate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                return_complex=True,
            )
        
        # Compute magnitude spectra
        noisy_mag = torch.abs(noisy_stft)
        noise_mag = torch.abs(noise_stft) if noise_estimate is not None else noise_estimate
        
        # Wiener filtering
        enhanced_mag = self._wiener_filter(noisy_mag, noise_mag)
        
        # Reconstruct phase
        phase = torch.angle(noisy_stft)
        enhanced_stft = enhanced_mag * torch.exp(1j * phase)
        
        # ISTFT
        enhanced_signal = torch.istft(
            enhanced_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            length=noisy_signal.shape[-1],
        )
        
        return enhanced_signal
    
    def _estimate_noise(self, noisy_stft: torch.Tensor) -> torch.Tensor:
        """Estimate noise from noisy signal using first few frames."""
        # Use first 10% of frames as noise estimate
        n_frames = noisy_stft.shape[-1]
        n_noise_frames = max(1, n_frames // 10)
        
        noise_stft = noisy_stft[..., :n_noise_frames]
        noise_mag = torch.abs(noise_stft).mean(dim=-1, keepdim=True)
        
        return noise_mag
    
    def _wiener_filter(
        self,
        noisy_mag: torch.Tensor,
        noise_mag: torch.Tensor,
    ) -> torch.Tensor:
        """Apply Wiener filter."""
        # Estimate speech power spectrum
        speech_mag = noisy_mag - noise_mag
        speech_mag = torch.maximum(speech_mag, torch.zeros_like(speech_mag))
        
        # Wiener filter gain
        snr = speech_mag / (noise_mag + 1e-8)
        gain = snr / (snr + 1)
        
        # Apply gain
        enhanced_mag = gain * noisy_mag
        
        return enhanced_mag


class SpectralGating(nn.Module):
    """Spectral gating for speech enhancement."""
    
    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: Optional[int] = None,
        window: str = "hann",
        alpha: float = 0.1,
        beta: float = 0.01,
    ):
        super().__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.window = window
        self.alpha = alpha
        self.beta = beta
        
        # Initialize window
        if window == "hann":
            self.register_buffer("window", torch.hann_window(self.win_length))
        elif window == "hamming":
            self.register_buffer("window", torch.hamming_window(self.win_length))
        else:
            self.register_buffer("window", torch.ones(self.win_length))
    
    def forward(
        self,
        noisy_signal: torch.Tensor,
        noise_estimate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            noisy_signal: Noisy speech signal [B, T]
            noise_estimate: Noise estimate [B, T] or None for auto-estimation
            
        Returns:
            Enhanced speech signal [B, T]
        """
        # Compute STFT
        noisy_stft = torch.stft(
            noisy_signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        
        # Estimate noise if not provided
        if noise_estimate is None:
            noise_estimate = self._estimate_noise(noisy_stft)
        else:
            noise_stft = torch.stft(
                noise_estimate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                return_complex=True,
            )
        
        # Compute magnitude spectra
        noisy_mag = torch.abs(noisy_stft)
        noise_mag = torch.abs(noise_stft) if noise_estimate is not None else noise_estimate
        
        # Spectral gating
        enhanced_mag = self._spectral_gating(noisy_mag, noise_mag)
        
        # Reconstruct phase
        phase = torch.angle(noisy_stft)
        enhanced_stft = enhanced_mag * torch.exp(1j * phase)
        
        # ISTFT
        enhanced_signal = torch.istft(
            enhanced_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            length=noisy_signal.shape[-1],
        )
        
        return enhanced_signal
    
    def _estimate_noise(self, noisy_stft: torch.Tensor) -> torch.Tensor:
        """Estimate noise from noisy signal using first few frames."""
        # Use first 10% of frames as noise estimate
        n_frames = noisy_stft.shape[-1]
        n_noise_frames = max(1, n_frames // 10)
        
        noise_stft = noisy_stft[..., :n_noise_frames]
        noise_mag = torch.abs(noise_stft).mean(dim=-1, keepdim=True)
        
        return noise_mag
    
    def _spectral_gating(
        self,
        noisy_mag: torch.Tensor,
        noise_mag: torch.Tensor,
    ) -> torch.Tensor:
        """Apply spectral gating."""
        # Compute gate threshold
        threshold = self.alpha * noise_mag
        
        # Apply gate
        gate = torch.where(noisy_mag > threshold, 1.0, self.beta)
        enhanced_mag = gate * noisy_mag
        
        return enhanced_mag
