import random
import numpy as np
import torch

def set_seed(seed: int):
    """
    Set the random seed for reproducibility.
    """
    random.seed(seed)                      # Python built-in random module
    np.random.seed(seed)                   # NumPy random seed
    torch.manual_seed(seed)                # PyTorch CPU seed
    torch.cuda.manual_seed_all(seed)       # PyTorch GPU seed (if applicable)
    torch.backends.cudnn.deterministic = True  # Ensure deterministic CNN behavior
    torch.backends.cudnn.benchmark = False  # Disable CuDNN auto-tuning for consistency

def generate_seeds(num_seeds=5, seed_range=(0, 10_000), fixed_seed=42):
    """
    Generate a list of unique random seeds.
    
    Args:
        num_seeds (int): Number of seeds to generate.
        seed_range (tuple): Range of seed values (low, high).
        fixed_seed (int): Fixed seed to ensure reproducibility.
    
    Returns:
        list: A list of unique seed values.
    """
    np.random.seed(fixed_seed)  # Set NumPy seed for reproducibility
    return np.random.choice(range(seed_range[0], seed_range[1]), size=num_seeds, replace=False).tolist()
