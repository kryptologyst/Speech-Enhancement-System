"""Data generation script for synthetic speech enhancement dataset."""

import argparse
from pathlib import Path

from src.data.synthetic_dataset import create_synthetic_dataset


def main():
    """Generate synthetic speech enhancement dataset."""
    parser = argparse.ArgumentParser(description="Generate synthetic dataset")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate")
    parser.add_argument("--duration", type=float, default=4.0, help="Duration in seconds")
    parser.add_argument("--n_samples_train", type=int, default=1000, help="Training samples")
    parser.add_argument("--n_samples_val", type=int, default=200, help="Validation samples")
    parser.add_argument("--n_samples_test", type=int, default=200, help="Test samples")
    parser.add_argument("--snr_range", type=float, nargs=2, default=[0, 20], help="SNR range")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Create data directory
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating synthetic dataset in {data_dir}")
    print(f"Training samples: {args.n_samples_train}")
    print(f"Validation samples: {args.n_samples_val}")
    print(f"Test samples: {args.n_samples_test}")
    print(f"SNR range: {args.snr_range[0]} - {args.snr_range[1]} dB")
    
    # Generate datasets
    train_dataset, val_dataset, test_dataset = create_synthetic_dataset(
        data_dir=data_dir,
        sample_rate=args.sample_rate,
        duration=args.duration,
        n_samples_train=args.n_samples_train,
        n_samples_val=args.n_samples_val,
        n_samples_test=args.n_samples_test,
        snr_range=tuple(args.snr_range),
        seed=args.seed,
    )
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Train: {len(train_dataset)} samples")
    print(f"Validation: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")
    
    print("\nTraining set statistics:")
    train_stats = train_dataset.get_statistics()
    for key, value in train_stats.items():
        print(f"  {key}: {value}")
    
    print("\nDataset generation completed!")


if __name__ == "__main__":
    main()
