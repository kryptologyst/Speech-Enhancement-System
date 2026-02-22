"""Utility functions for device management and deterministic behavior."""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> torch.device:
    """Get the best available device with fallback chain.
    
    Args:
        device: Preferred device ('auto', 'cpu', 'cuda', 'mps')
        
    Returns:
        torch.device: The selected device
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def get_device_info() -> dict:
    """Get information about the current device.
    
    Returns:
        dict: Device information including name, memory, etc.
    """
    device = get_device()
    info = {"device": str(device)}
    
    if device.type == "cuda":
        info.update({
            "name": torch.cuda.get_device_name(0),
            "memory_total": torch.cuda.get_device_properties(0).total_memory,
            "memory_allocated": torch.cuda.memory_allocated(0),
            "memory_reserved": torch.cuda.memory_reserved(0),
        })
    elif device.type == "mps":
        info["name"] = "Apple Silicon GPU"
    else:
        info["name"] = "CPU"
        
    return info


def move_to_device(tensor_or_dict, device: torch.device):
    """Move tensor or dictionary of tensors to device.
    
    Args:
        tensor_or_dict: Tensor or dict of tensors
        device: Target device
        
    Returns:
        Tensor or dict moved to device
    """
    if isinstance(tensor_or_dict, torch.Tensor):
        return tensor_or_dict.to(device)
    elif isinstance(tensor_or_dict, dict):
        return {k: move_to_device(v, device) for k, v in tensor_or_dict.items()}
    elif isinstance(tensor_or_dict, (list, tuple)):
        return type(tensor_or_dict)(move_to_device(item, device) for item in tensor_or_dict)
    else:
        return tensor_or_dict
