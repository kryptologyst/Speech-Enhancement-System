"""Basic tests for speech enhancement system."""

import pytest
import torch
import numpy as np

from src.models.conv_tasnet import ConvTasNet
from src.models.spectral_methods import SpectralSubtraction, WienerFilter
from src.utils.audio import generate_synthetic_speech, generate_noise, add_noise
from src.metrics.enhancement import EnhancementMetrics


def test_conv_tasnet():
    """Test Conv-TasNet model."""
    model = ConvTasNet()
    
    # Test input
    batch_size = 2
    duration = 4.0
    sample_rate = 16000
    n_samples = int(duration * sample_rate)
    
    noisy_input = torch.randn(batch_size, 1, n_samples)
    
    # Forward pass
    enhanced, separated = model(noisy_input)
    
    # Check output shapes
    assert enhanced.shape == (batch_size, 1, n_samples)
    assert separated is None  # Only 2 sources, so separated is None
    
    # Check model size
    assert model.get_model_size() > 0
    assert model.get_model_size_mb() > 0


def test_spectral_methods():
    """Test spectral enhancement methods."""
    # Generate test signal
    clean_speech = generate_synthetic_speech(duration=2.0, sample_rate=16000)
    noise = generate_noise(duration=2.0, sample_rate=16000, noise_type="white")
    noisy_speech = add_noise(clean_speech, noise, snr_db=10)
    
    # Convert to tensors
    noisy_tensor = torch.from_numpy(noisy_speech).float()
    
    # Test Spectral Subtraction
    ss_model = SpectralSubtraction()
    enhanced_ss = ss_model(noisy_tensor)
    assert enhanced_ss.shape == noisy_tensor.shape
    
    # Test Wiener Filter
    wf_model = WienerFilter()
    enhanced_wf = wf_model(noisy_tensor)
    assert enhanced_wf.shape == noisy_tensor.shape


def test_enhancement_metrics():
    """Test enhancement metrics."""
    # Generate test signals
    clean = torch.randn(2, 16000)
    enhanced = clean + 0.1 * torch.randn(2, 16000)
    noisy = clean + 0.5 * torch.randn(2, 16000)
    
    # Test metrics
    metrics = EnhancementMetrics()
    
    # Test individual metrics
    si_sdr_val = metrics.compute_all(enhanced, clean, noisy)
    assert "si_sdr" in si_sdr_val
    assert "sdr" in si_sdr_val
    assert "snr" in si_sdr_val
    
    # Test batch average
    avg_metrics = metrics.compute_batch_average(enhanced, clean, noisy)
    assert isinstance(avg_metrics, dict)
    assert len(avg_metrics) > 0


def test_audio_utilities():
    """Test audio utility functions."""
    # Test synthetic speech generation
    speech = generate_synthetic_speech(duration=1.0, sample_rate=16000)
    assert len(speech) == 16000
    assert np.all(np.isfinite(speech))
    
    # Test noise generation
    noise = generate_noise(duration=1.0, sample_rate=16000, noise_type="white")
    assert len(noise) == 16000
    assert np.all(np.isfinite(noise))
    
    # Test noise addition
    noisy = add_noise(speech, noise, snr_db=10)
    assert len(noisy) == 16000
    assert np.all(np.isfinite(noisy))


def test_device_utilities():
    """Test device utility functions."""
    from src.utils.device import get_device, set_seed
    
    # Test device detection
    device = get_device("auto")
    assert isinstance(device, torch.device)
    
    # Test seeding
    set_seed(42)
    # Should not raise any errors


if __name__ == "__main__":
    pytest.main([__file__])
