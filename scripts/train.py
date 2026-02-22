"""Training script for speech enhancement models."""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.synthetic_dataset import create_synthetic_dataset
from src.metrics.enhancement import EnhancementMetrics
from src.models.conv_tasnet import ConvTasNet
from src.models.spectral_methods import SpectralSubtraction, WienerFilter
from src.utils.device import get_device, move_to_device, set_seed


def setup_logging(log_dir: Path, log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "training.log"),
            logging.StreamHandler(),
        ],
    )


def create_model(model_type: str, **kwargs) -> nn.Module:
    """Create model based on type."""
    if model_type == "conv_tasnet":
        return ConvTasNet(**kwargs)
    elif model_type == "spectral_subtraction":
        return SpectralSubtraction(**kwargs)
    elif model_type == "wiener_filter":
        return WienerFilter(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_loss_function(loss_type: str) -> nn.Module:
    """Create loss function."""
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "l1":
        return nn.L1Loss()
    elif loss_type == "smooth_l1":
        return nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        batch = move_to_device(batch, device)
        
        # Get data
        clean = batch["clean"]
        noisy = batch["noisy"]
        
        # Forward pass
        optimizer.zero_grad()
        
        if isinstance(model, ConvTasNet):
            enhanced, _ = model(noisy.unsqueeze(1))
            enhanced = enhanced.squeeze(1)
        else:
            enhanced = model(noisy)
        
        # Compute loss
        loss = criterion(enhanced, clean)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item() * clean.size(0)
        total_samples += clean.size(0)
        
        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / total_samples
    return {"train_loss": avg_loss}


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    metrics: EnhancementMetrics,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()
    
    total_loss = 0.0
    total_samples = 0
    
    all_metrics = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Validation {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch = move_to_device(batch, device)
            
            # Get data
            clean = batch["clean"]
            noisy = batch["noisy"]
            
            # Forward pass
            if isinstance(model, ConvTasNet):
                enhanced, _ = model(noisy.unsqueeze(1))
                enhanced = enhanced.squeeze(1)
            else:
                enhanced = model(noisy)
            
            # Compute loss
            loss = criterion(enhanced, clean)
            
            # Compute metrics
            batch_metrics = metrics.compute_batch_average(enhanced, clean, noisy)
            all_metrics.append(batch_metrics)
            
            # Update statistics
            total_loss += loss.item() * clean.size(0)
            total_samples += clean.size(0)
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / total_samples
    
    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
    
    return {"val_loss": avg_loss, **avg_metrics}


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_dir: Path,
    is_best: bool = False,
) -> None:
    """Save model checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    
    # Save latest checkpoint
    torch.save(checkpoint, checkpoint_dir / "latest.pth")
    
    # Save best checkpoint
    if is_best:
        torch.save(checkpoint, checkpoint_dir / "best.pth")


def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    checkpoint_path: Path,
) -> int:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint["epoch"]


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train speech enhancement model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--log_dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--model_type", type=str, default="conv_tasnet", help="Model type")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = get_device(args.device)
    
    # Create directories
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    log_dir = Path(args.log_dir)
    
    setup_logging(log_dir)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Training on device: {device}")
    logger.info(f"Model type: {args.model_type}")
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset, val_dataset, test_dataset = create_synthetic_dataset(
        data_dir=args.data_dir,
        seed=args.seed,
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # Create model
    logger.info("Creating model...")
    if args.model_type == "conv_tasnet":
        model = ConvTasNet()
    elif args.model_type == "spectral_subtraction":
        model = SpectralSubtraction()
    elif args.model_type == "wiener_filter":
        model = WienerFilter()
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    model = model.to(device)
    
    # Create optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = create_loss_function("mse")
    
    # Create metrics
    metrics = EnhancementMetrics()
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_metric = float("inf")
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch = load_checkpoint(model, optimizer, Path(args.resume))
    
    # Training loop
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, metrics, device, epoch)
        
        # Log metrics
        logger.info(f"Epoch {epoch}: {train_metrics} | {val_metrics}")
        
        # Save checkpoint
        is_best = val_metrics["val_loss"] < best_metric
        if is_best:
            best_metric = val_metrics["val_loss"]
        
        save_checkpoint(model, optimizer, epoch, val_metrics, checkpoint_dir, is_best)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
