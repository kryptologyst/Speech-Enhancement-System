"""Audio processing utilities."""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio

logger = logging.getLogger(__name__)


def load_audio(
    file_path: Union[str, Path],
    sample_rate: Optional[int] = None,
    mono: bool = True,
    normalize: bool = True,
) -> Tuple[np.ndarray, int]:
    """Load audio file with librosa.
    
    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate (None for original)
        mono: Convert to mono
        normalize: Normalize audio to [-1, 1]
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    try:
        audio, sr = librosa.load(
            str(file_path),
            sr=sample_rate,
            mono=mono,
            dtype=np.float32
        )
        
        if normalize:
            audio = librosa.util.normalize(audio)
            
        return audio, sr
    except Exception as e:
        logger.error(f"Error loading audio {file_path}: {e}")
        raise


def save_audio(
    audio: np.ndarray,
    file_path: Union[str, Path],
    sample_rate: int,
    format: str = "WAV",
    subtype: str = "PCM_16",
) -> None:
    """Save audio array to file.
    
    Args:
        audio: Audio array
        file_path: Output file path
        sample_rate: Sample rate
        format: Audio format
        subtype: Audio subtype
    """
    try:
        sf.write(
            str(file_path),
            audio,
            sample_rate,
            format=format,
            subtype=subtype
        )
    except Exception as e:
        logger.error(f"Error saving audio {file_path}: {e}")
        raise


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample audio to target sample rate.
    
    Args:
        audio: Input audio array
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled audio array
    """
    if orig_sr == target_sr:
        return audio
        
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def compute_stft(
    audio: np.ndarray,
    n_fft: int = 512,
    hop_length: int = 256,
    win_length: Optional[int] = None,
    window: str = "hann",
) -> np.ndarray:
    """Compute Short-Time Fourier Transform.
    
    Args:
        audio: Input audio array
        n_fft: FFT window size
        hop_length: Hop length
        win_length: Window length
        window: Window type
        
    Returns:
        STFT complex array
    """
    return librosa.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
    )


def compute_istft(
    stft: np.ndarray,
    hop_length: int = 256,
    win_length: Optional[int] = None,
    window: str = "hann",
    length: Optional[int] = None,
) -> np.ndarray:
    """Compute Inverse Short-Time Fourier Transform.
    
    Args:
        stft: STFT complex array
        hop_length: Hop length
        win_length: Window length
        window: Window type
        length: Output length
        
    Returns:
        Reconstructed audio array
    """
    return librosa.istft(
        stft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=length,
    )


def compute_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 512,
    hop_length: int = 256,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> np.ndarray:
    """Compute mel spectrogram.
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate
        n_fft: FFT window size
        hop_length: Hop length
        n_mels: Number of mel bins
        fmin: Minimum frequency
        fmax: Maximum frequency
        
    Returns:
        Mel spectrogram array
    """
    if fmax is None:
        fmax = sample_rate // 2
        
    return librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )


def add_noise(
    clean_audio: np.ndarray,
    noise_audio: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """Add noise to clean audio at specified SNR.
    
    Args:
        clean_audio: Clean speech signal
        noise_audio: Noise signal
        snr_db: Signal-to-noise ratio in dB
        
    Returns:
        Noisy audio array
    """
    # Ensure noise is same length as clean audio
    if len(noise_audio) > len(clean_audio):
        noise_audio = noise_audio[:len(clean_audio)]
    elif len(noise_audio) < len(clean_audio):
        # Repeat noise if shorter
        repeats = len(clean_audio) // len(noise_audio) + 1
        noise_audio = np.tile(noise_audio, repeats)[:len(clean_audio)]
    
    # Calculate noise scaling factor
    signal_power = np.mean(clean_audio ** 2)
    noise_power = np.mean(noise_audio ** 2)
    snr_linear = 10 ** (snr_db / 10)
    
    noise_scaling = np.sqrt(signal_power / (noise_power * snr_linear))
    scaled_noise = noise_audio * noise_scaling
    
    return clean_audio + scaled_noise


def generate_synthetic_speech(
    duration: float,
    sample_rate: int,
    freq_range: Tuple[float, float] = (100, 8000),
) -> np.ndarray:
    """Generate synthetic speech-like signal.
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate
        freq_range: Frequency range (min, max) in Hz
        
    Returns:
        Synthetic speech signal
    """
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples)
    
    # Generate multiple harmonics with varying amplitudes
    signal = np.zeros(n_samples)
    base_freq = np.random.uniform(freq_range[0], freq_range[1] // 4)
    
    for harmonic in range(1, 8):
        freq = base_freq * harmonic
        if freq > freq_range[1]:
            break
            
        amplitude = 1.0 / harmonic
        phase = np.random.uniform(0, 2 * np.pi)
        signal += amplitude * np.sin(2 * np.pi * freq * t + phase)
    
    # Add some modulation
    modulation = 0.1 * np.sin(2 * np.pi * 5 * t)
    signal *= (1 + modulation)
    
    # Normalize
    signal = librosa.util.normalize(signal)
    
    return signal


def generate_noise(
    duration: float,
    sample_rate: int,
    noise_type: str = "white",
) -> np.ndarray:
    """Generate synthetic noise.
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate
        noise_type: Type of noise ('white', 'pink', 'brown')
        
    Returns:
        Noise signal
    """
    n_samples = int(duration * sample_rate)
    
    if noise_type == "white":
        noise = np.random.randn(n_samples)
    elif noise_type == "pink":
        # Pink noise (1/f)
        white = np.random.randn(n_samples)
        freqs = np.fft.fftfreq(n_samples)
        freqs[0] = 1  # Avoid division by zero
        pink_filter = 1 / np.sqrt(np.abs(freqs))
        noise = np.real(np.fft.ifft(np.fft.fft(white) * pink_filter))
    elif noise_type == "brown":
        # Brown noise (1/f^2)
        white = np.random.randn(n_samples)
        freqs = np.fft.fftfreq(n_samples)
        freqs[0] = 1
        brown_filter = 1 / np.abs(freqs)
        noise = np.real(np.fft.ifft(np.fft.fft(white) * brown_filter))
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return librosa.util.normalize(noise)
