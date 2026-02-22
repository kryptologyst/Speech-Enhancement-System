"""Evaluation metrics for speech enhancement."""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

try:
    import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False

try:
    import pystoi
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False


def si_sdr(
    estimate: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
    
    Args:
        estimate: Estimated signal [B, T]
        target: Target signal [B, T]
        eps: Small constant for numerical stability
        
    Returns:
        SI-SDR values [B]
    """
    # Ensure same length
    min_len = min(estimate.shape[-1], target.shape[-1])
    estimate = estimate[..., :min_len]
    target = target[..., :min_len]
    
    # Zero-mean
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    
    # Compute optimal scaling
    dot_product = (estimate * target).sum(dim=-1, keepdim=True)
    target_energy = (target * target).sum(dim=-1, keepdim=True)
    scaling = dot_product / (target_energy + eps)
    
    # Scaled target
    scaled_target = scaling * target
    
    # SI-SDR
    signal_power = (scaled_target * scaled_target).sum(dim=-1)
    noise_power = ((estimate - scaled_target) * (estimate - scaled_target)).sum(dim=-1)
    si_sdr = 10 * torch.log10(signal_power / (noise_power + eps))
    
    return si_sdr


def sdr(
    estimate: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Signal-to-Distortion Ratio (SDR).
    
    Args:
        estimate: Estimated signal [B, T]
        target: Target signal [B, T]
        eps: Small constant for numerical stability
        
    Returns:
        SDR values [B]
    """
    # Ensure same length
    min_len = min(estimate.shape[-1], target.shape[-1])
    estimate = estimate[..., :min_len]
    target = target[..., :min_len]
    
    # SDR
    signal_power = (target * target).sum(dim=-1)
    noise_power = ((estimate - target) * (estimate - target)).sum(dim=-1)
    sdr = 10 * torch.log10(signal_power / (noise_power + eps))
    
    return sdr


def snr(
    estimate: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Signal-to-Noise Ratio (SNR).
    
    Args:
        estimate: Estimated signal [B, T]
        target: Target signal [B, T]
        eps: Small constant for numerical stability
        
    Returns:
        SNR values [B]
    """
    # Ensure same length
    min_len = min(estimate.shape[-1], target.shape[-1])
    estimate = estimate[..., :min_len]
    target = target[..., :min_len]
    
    # SNR
    signal_power = (target * target).sum(dim=-1)
    noise_power = ((estimate - target) * (estimate - target)).sum(dim=-1)
    snr = 10 * torch.log10(signal_power / (noise_power + eps))
    
    return snr


def pesq_score(
    estimate: torch.Tensor,
    target: torch.Tensor,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """PESQ (Perceptual Evaluation of Speech Quality) score.
    
    Args:
        estimate: Estimated signal [B, T]
        target: Target signal [B, T]
        sample_rate: Sample rate
        
    Returns:
        PESQ scores [B]
    """
    if not PESQ_AVAILABLE:
        raise ImportError("pesq package not available")
    
    # Convert to numpy
    estimate_np = estimate.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    # Compute PESQ for each sample
    pesq_scores = []
    for i in range(estimate_np.shape[0]):
        try:
            score = pesq.pesq(sample_rate, target_np[i], estimate_np[i], "wb")
            pesq_scores.append(score)
        except:
            pesq_scores.append(1.0)  # Fallback score
    
    return torch.tensor(pesq_scores, device=estimate.device)


def stoi_score(
    estimate: torch.Tensor,
    target: torch.Tensor,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """STOI (Short-Time Objective Intelligibility) score.
    
    Args:
        estimate: Estimated signal [B, T]
        target: Target signal [B, T]
        sample_rate: Sample rate
        
    Returns:
        STOI scores [B]
    """
    if not STOI_AVAILABLE:
        raise ImportError("pystoi package not available")
    
    # Convert to numpy
    estimate_np = estimate.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    # Compute STOI for each sample
    stoi_scores = []
    for i in range(estimate_np.shape[0]):
        try:
            score = pystoi.stoi(target_np[i], estimate_np[i], sample_rate, extended=False)
            stoi_scores.append(score)
        except:
            stoi_scores.append(0.0)  # Fallback score
    
    return torch.tensor(stoi_scores, device=estimate.device)


class EnhancementMetrics:
    """Comprehensive metrics for speech enhancement evaluation."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        compute_pesq: bool = True,
        compute_stoi: bool = True,
    ):
        self.sample_rate = sample_rate
        self.compute_pesq = compute_pesq and PESQ_AVAILABLE
        self.compute_stoi = compute_stoi and STOI_AVAILABLE
    
    def compute_all(
        self,
        estimate: torch.Tensor,
        target: torch.Tensor,
        mixture: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute all available metrics.
        
        Args:
            estimate: Estimated signal [B, T]
            target: Target signal [B, T]
            mixture: Mixture signal [B, T] (optional)
            
        Returns:
            Dictionary of metric names and values
        """
        metrics = {}
        
        # Basic metrics
        metrics["si_sdr"] = si_sdr(estimate, target)
        metrics["sdr"] = sdr(estimate, target)
        metrics["snr"] = snr(estimate, target)
        
        # Improvement metrics
        if mixture is not None:
            metrics["si_sdr_improvement"] = si_sdr(estimate, target) - si_sdr(mixture, target)
            metrics["sdr_improvement"] = sdr(estimate, target) - sdr(mixture, target)
            metrics["snr_improvement"] = snr(estimate, target) - snr(mixture, target)
        
        # Perceptual metrics
        if self.compute_pesq:
            metrics["pesq"] = pesq_score(estimate, target, self.sample_rate)
        
        if self.compute_stoi:
            metrics["stoi"] = stoi_score(estimate, target, self.sample_rate)
        
        return metrics
    
    def compute_batch_average(
        self,
        estimate: torch.Tensor,
        target: torch.Tensor,
        mixture: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Compute batch-averaged metrics.
        
        Args:
            estimate: Estimated signal [B, T]
            target: Target signal [B, T]
            mixture: Mixture signal [B, T] (optional)
            
        Returns:
            Dictionary of metric names and average values
        """
        metrics = self.compute_all(estimate, target, mixture)
        return {k: v.mean().item() for k, v in metrics.items()}
    
    def compute_individual(
        self,
        estimate: torch.Tensor,
        target: torch.Tensor,
        mixture: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute individual sample metrics.
        
        Args:
            estimate: Estimated signal [B, T]
            target: Target signal [B, T]
            mixture: Mixture signal [B, T] (optional)
            
        Returns:
            Dictionary of metric names and individual values
        """
        return self.compute_all(estimate, target, mixture)
