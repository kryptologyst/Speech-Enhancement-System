"""Evaluation script for speech enhancement models."""

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from src.data.synthetic_dataset import SyntheticDataset
from src.metrics.enhancement import EnhancementMetrics
from src.models.conv_tasnet import ConvTasNet
from src.models.spectral_methods import SpectralSubtraction, WienerFilter
from src.utils.device import get_device, move_to_device


def load_model(model_path: Path, model_type: str, device: torch.device) -> torch.nn.Module:
    """Load trained model."""
    if model_type == "conv_tasnet":
        model = ConvTasNet()
    elif model_type == "spectral_subtraction":
        model = SpectralSubtraction()
    elif model_type == "wiener_filter":
        model = WienerFilter()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    metrics: EnhancementMetrics,
    device: torch.device,
    model_type: str,
) -> Dict[str, float]:
    """Evaluate model on dataset."""
    model.eval()
    
    all_metrics = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            batch = move_to_device(batch, device)
            
            # Get data
            clean = batch["clean"]
            noisy = batch["noisy"]
            
            # Forward pass
            if model_type == "conv_tasnet":
                enhanced, _ = model(noisy.unsqueeze(1))
                enhanced = enhanced.squeeze(1)
            else:
                enhanced = model(noisy)
            
            # Compute metrics
            batch_metrics = metrics.compute_batch_average(enhanced, clean, noisy)
            all_metrics.append(batch_metrics)
    
    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
    
    return avg_metrics


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate speech enhancement model")
    parser.add_argument("--model_path", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--model_type", type=str, required=True, help="Model type")
    parser.add_argument("--test_data", type=str, required=True, help="Test data directory")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    
    args = parser.parse_args()
    
    # Setup
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Evaluating model: {args.model_path}")
    print(f"Model type: {args.model_type}")
    print(f"Device: {device}")
    
    # Load model
    model = load_model(Path(args.model_path), args.model_type, device)
    
    # Load test dataset
    test_dataset = SyntheticDataset(
        data_dir=Path(args.test_data),
        split="test",
        generate_on_init=False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # Create metrics
    metrics = EnhancementMetrics()
    
    # Evaluate
    print("Evaluating model...")
    results = evaluate_model(model, test_loader, metrics, device, args.model_type)
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    for metric, value in results.items():
        print(f"{metric:20s}: {value:8.4f}")
    
    # Save results
    results_file = output_dir / f"{args.model_type}_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
