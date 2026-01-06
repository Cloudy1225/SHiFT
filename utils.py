"""
utils.py

Utility functions for hierarchical TAG learning
"""
import os
import random
import numpy as np
import torch
from datetime import datetime


# Model paths for text encoders
MODEL_PATHs = {
    # LM
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "SentenceBert": "sentence-transformers/multi-qa-distilbert-cos-v1",
    "e5-large": "intfloat/e5-large-v2",
    "roberta": "sentence-transformers/all-roberta-large-v1",

    # LLM
    "Qwen-3B": "Qwen/Qwen2.5-3B-Instruct",
    "Qwen-7B": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen-14B": "Qwen/Qwen2.5-14B-Instruct",
    "Qwen-32B": "Qwen/Qwen2.5-32B-Instruct",
    "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.2",
    "Llama-8B": "meta-llama/Llama-3.1-8B-Instruct",
}


def set_random_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cur_time(timezone='Asia/Shanghai', t_format='%m-%d %H:%M:%S'):
    """Get current time string"""
    from datetime import datetime
    import pytz

    try:
        tz = pytz.timezone(timezone)
        return datetime.now(tz).strftime(t_format)
    except:
        return datetime.now().strftime(t_format)


class EarlyStopping:
    """Early stopping to stop training when loss doesn't improve"""

    def __init__(self, patience=20, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last improvement (default: 20)
            verbose (bool): If True, prints a message for each improvement (default: False)
            delta (float): Minimum change to qualify as improvement (default: 0)
            path (str): Path for checkpoint (default: 'checkpoint.pt')
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model=None):
        """
        Args:
            val_loss (float): Validation loss
            model: Model to save checkpoint (optional)
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            if model is not None:
                self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if model is not None:
                self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Save model when validation loss decreases"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_accuracy(pred, labels):
    """Compute classification accuracy"""
    return (pred.argmax(dim=1) == labels).float().mean().item()


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """Save checkpoint to file"""
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None):
    """Load checkpoint from file"""
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"No checkpoint found at '{filename}'")

    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint.get('epoch', 0)


def count_parameters(model):
    """Count number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model):
    """Print model architecture and parameter count"""
    print("=" * 60)
    print("Model Architecture:")
    print("=" * 60)
    print(model)
    print("=" * 60)
    print(f"Total Parameters: {count_parameters(model):,}")
    print("=" * 60)
