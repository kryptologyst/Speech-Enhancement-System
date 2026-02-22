"""Synthetic dataset for speech enhancement."""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.audio import (
    add_noise,
    generate_noise,
    generate_synthetic_speech,
    save_audio,
)


class SyntheticDataset(Dataset):
    """Synthetic dataset for speech enhancement training."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        sample_rate: int = 16000,
        duration: float = 4.0,
        n_samples: int = 1000,
        speech_freq_range: Tuple[float, float] = (100, 8000),
        noise_types: List[str] = None,
        snr_range: Tuple[float, float] = (0, 20),
        n_fft: int = 512,
        hop_length: int = 256,
        n_mels: int = 80,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        split: str = "train",
        seed: Optional[int] = None,
        generate_on_init: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = n_samples
        self.speech_freq_range = speech_freq_range
        self.noise_types = noise_types or ["white", "pink", "brown"]
        self.snr_range = snr_range
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or sample_rate // 2
        self.split = split
        self.seed = seed
        
        # Set random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate data if requested
        if generate_on_init:
            self._generate_data()
        
        # Load metadata
        self.metadata = self._load_metadata()
    
    def _generate_data(self) -> None:
        """Generate synthetic speech enhancement dataset."""
        print(f"Generating {self.n_samples} samples for {self.split} split...")
        
        metadata = []
        
        for i in range(self.n_samples):
            # Generate clean speech
            clean_speech = generate_synthetic_speech(
                duration=self.duration,
                sample_rate=self.sample_rate,
                freq_range=self.speech_freq_range,
            )
            
            # Generate noise
            noise_type = random.choice(self.noise_types)
            noise = generate_noise(
                duration=self.duration,
                sample_rate=self.sample_rate,
                noise_type=noise_type,
            )
            
            # Generate SNR
            snr_db = random.uniform(*self.snr_range)
            
            # Create noisy mixture
            noisy_mixture = add_noise(clean_speech, noise, snr_db)
            
            # Save audio files
            sample_id = f"{self.split}_{i:06d}"
            clean_path = self.data_dir / f"{sample_id}_clean.wav"
            noisy_path = self.data_dir / f"{sample_id}_noisy.wav"
            noise_path = self.data_dir / f"{sample_id}_noise.wav"
            
            save_audio(clean_speech, clean_path, self.sample_rate)
            save_audio(noisy_mixture, noisy_path, self.sample_rate)
            save_audio(noise, noise_path, self.sample_rate)
            
            # Store metadata
            metadata.append({
                "sample_id": sample_id,
                "clean_path": str(clean_path),
                "noisy_path": str(noisy_path),
                "noise_path": str(noise_path),
                "noise_type": noise_type,
                "snr_db": snr_db,
                "duration": self.duration,
                "sample_rate": self.sample_rate,
            })
        
        # Save metadata
        metadata_df = pd.DataFrame(metadata)
        metadata_path = self.data_dir / f"{self.split}_metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)
        
        print(f"Generated {self.n_samples} samples. Metadata saved to {metadata_path}")
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load dataset metadata."""
        metadata_path = self.data_dir / f"{self.split}_metadata.csv"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        return pd.read_csv(metadata_path)
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item."""
        row = self.metadata.iloc[idx]
        
        # Load audio files
        clean_path = Path(row["clean_path"])
        noisy_path = Path(row["noisy_path"])
        noise_path = Path(row["noise_path"])
        
        # Load audio
        clean_audio, _ = self._load_audio(clean_path)
        noisy_audio, _ = self._load_audio(noisy_path)
        noise_audio, _ = self._load_audio(noise_path)
        
        # Convert to tensors
        clean_tensor = torch.from_numpy(clean_audio).float()
        noisy_tensor = torch.from_numpy(noisy_audio).float()
        noise_tensor = torch.from_numpy(noise_audio).float()
        
        return {
            "clean": clean_tensor,
            "noisy": noisy_tensor,
            "noise": noise_tensor,
            "sample_id": row["sample_id"],
            "noise_type": row["noise_type"],
            "snr_db": row["snr_db"],
        }
    
    def _load_audio(self, path: Path) -> Tuple[np.ndarray, int]:
        """Load audio file."""
        import librosa
        
        audio, sr = librosa.load(str(path), sr=self.sample_rate, mono=True)
        return audio, sr
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get sample information."""
        return self.metadata.iloc[idx].to_dict()
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            "n_samples": len(self.metadata),
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "noise_types": self.metadata["noise_type"].unique().tolist(),
            "snr_range": [self.metadata["snr_db"].min(), self.metadata["snr_db"].max()],
            "snr_mean": self.metadata["snr_db"].mean(),
            "snr_std": self.metadata["snr_db"].std(),
        }
        
        return stats


def create_synthetic_dataset(
    data_dir: Union[str, Path],
    sample_rate: int = 16000,
    duration: float = 4.0,
    n_samples_train: int = 1000,
    n_samples_val: int = 200,
    n_samples_test: int = 200,
    speech_freq_range: Tuple[float, float] = (100, 8000),
    noise_types: List[str] = None,
    snr_range: Tuple[float, float] = (0, 20),
    seed: int = 42,
) -> Tuple[SyntheticDataset, SyntheticDataset, SyntheticDataset]:
    """Create train/val/test synthetic datasets.
    
    Args:
        data_dir: Data directory
        sample_rate: Sample rate
        duration: Duration in seconds
        n_samples_train: Number of training samples
        n_samples_val: Number of validation samples
        n_samples_test: Number of test samples
        speech_freq_range: Speech frequency range
        noise_types: Types of noise
        snr_range: SNR range in dB
        seed: Random seed
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    data_dir = Path(data_dir)
    
    # Create datasets
    train_dataset = SyntheticDataset(
        data_dir=data_dir / "train",
        sample_rate=sample_rate,
        duration=duration,
        n_samples=n_samples_train,
        speech_freq_range=speech_freq_range,
        noise_types=noise_types,
        snr_range=snr_range,
        split="train",
        seed=seed,
    )
    
    val_dataset = SyntheticDataset(
        data_dir=data_dir / "val",
        sample_rate=sample_rate,
        duration=duration,
        n_samples=n_samples_val,
        speech_freq_range=speech_freq_range,
        noise_types=noise_types,
        snr_range=snr_range,
        split="val",
        seed=seed + 1,
    )
    
    test_dataset = SyntheticDataset(
        data_dir=data_dir / "test",
        sample_rate=sample_rate,
        duration=duration,
        n_samples=n_samples_test,
        speech_freq_range=speech_freq_range,
        noise_types=noise_types,
        snr_range=snr_range,
        split="test",
        seed=seed + 2,
    )
    
    return train_dataset, val_dataset, test_dataset
