import torch
from torch.utils.data import Dataset, TensorDataset
from typing import Optional
import numpy as np


class SlidingWindowDataset(Dataset):
    '''
    Creates a sliding window dataset from a sequence of magtense simulations.
    Generated with chatGPT
    '''
    def __init__(self, X, U, window_size=32, step=1) -> None:
        """
        X, Y: torch.Tensor of shape (n_seq, seq_length, H, W, C)
        U: torch.Tensor of shape (n_seq, seq_length, action_dim)
        window_size: length of each subsequence
        step: sliding window step size
        """
        n_seq, seq_len = X.shape[:2]
        self.indices = []
        for seq_idx in range(n_seq):
            for start in range(0, seq_len - window_size + 1, step):
                self.indices.append((seq_idx, start))
        self.X = X
        self.U = U
        self.window_size = window_size

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_idx, start = self.indices[idx]
        end = start + self.window_size
        x_window = self.X[seq_idx, start:end]  # (window_size, H, W, C)
        u_window = self.U[seq_idx, start:end]  # (window_size, action_dim)
        return x_window, u_window


def train_test_split(X : np.ndarray, 
                     Hs : np.ndarray, 
                     dataset_type : str = "window",
                     test_size : float = 0.2, 
                     window_size : int = 32,
                     step : int = 1,
                     seed : Optional[int] = None,
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Splits the dataset into training and testing sets.
    X, U: torch.Tensor of shape (n_seq, seq_length, H, W, C) and (n_seq, seq_length, action_dim)
    test_size: proportion of the dataset to include in the test split
    seed: random seed for reproducibility
    """
    if seed is not None:
        torch.manual_seed(seed)
        
    X = torch.tensor(X).to(torch.float32)

    U = torch.stack([
    torch.stack([torch.tensor(Hs[i]).to(torch.float32) for _ in range(X.shape[1])])
    for i in range(X.shape[0])])
    
    n_seq = X.shape[0]
    indices = torch.randperm(n_seq)
    split = int(n_seq * (1 - test_size))
    
    train_indices = indices[:split]
    test_indices = indices[split:]
    
    X_train = X[train_indices]
    U_train = U[train_indices]
    
    X_test = X[test_indices]
    U_test = U[test_indices]

    if dataset_type == "window":
        train_dataset = SlidingWindowDataset(X_train, U_train, window_size=window_size, step=step)
        test_dataset = SlidingWindowDataset(X_test, U_test, window_size=window_size, step=step)
    
    elif dataset_type == "full":
        train_dataset = TensorDataset(X_train, U_train)
        test_dataset = TensorDataset(X_test, U_test)
    
    return train_dataset, test_dataset, train_indices, test_indices



import torch
from torch.utils.data import Dataset, TensorDataset
from typing import Optional
import numpy as np


class SlidingWindowDataset(Dataset):
    '''
    Creates a sliding window dataset from a sequence of magtense simulations.
    Generated with chatGPT
    '''
    def __init__(self, X, U, window_size=32, step=1) -> None:
        """
        X, Y: torch.Tensor of shape (n_seq, seq_length, H, W, C)
        U: torch.Tensor of shape (n_seq, seq_length, action_dim)
        window_size: length of each subsequence
        step: sliding window step size
        """
        n_seq, seq_len = X.shape[:2]
        self.indices = []
        for seq_idx in range(n_seq):
            for start in range(0, seq_len - window_size + 1, step):
                self.indices.append((seq_idx, start))
        self.X = X
        self.U = U
        self.window_size = window_size

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_idx, start = self.indices[idx]
        end = start + self.window_size
        x_window = self.X[seq_idx, start:end]  # (window_size, H, W, C)
        u_window = self.U[seq_idx, start:end]  # (window_size, action_dim)
        return x_window, u_window


def train_test_val_split(X : np.ndarray, 
                        Hs : np.ndarray, 
                        dataset_type : str = "window",
                        test_size : float = 0.2, 
                        val_size : float = 0.2,
                        window_size : int = 32,
                        step : int = 1,
                        seed : Optional[int] = None,
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Splits the dataset into training and testing sets.
    X, U: torch.Tensor of shape (n_seq, seq_length, H, W, C) and (n_seq, seq_length, action_dim)
    test_size: proportion of the dataset to include in the test split
    seed: random seed for reproducibility
    """
    if seed is not None:
        torch.manual_seed(seed)
        
    X = torch.tensor(X).to(torch.float32)

    U = torch.stack([
    torch.stack([torch.tensor(Hs[i]).to(torch.float32) for _ in range(X.shape[1])])
    for i in range(X.shape[0])])
    
    n_seq = X.shape[0]
    indices = torch.randperm(n_seq)
    split1 = int(n_seq * (1 - test_size))
    split2 = int(split1 * (1 - val_size))
    
    tv_indices = indices[:split1]
    
    train_indices = tv_indices[:split2]
    val_indices = tv_indices[split2:]
    test_indices = indices[split1:]
    
    X_train = X[train_indices]
    U_train = U[train_indices]

    X_val = X[val_indices]
    U_val = U[val_indices]
    
    X_test = X[test_indices]
    U_test = U[test_indices]

    if dataset_type == "window":
        train_dataset = SlidingWindowDataset(X_train, U_train, window_size=window_size, step=step)
        val_dataset = SlidingWindowDataset(X_val, U_val, window_size=window_size, step=step)
        test_dataset = SlidingWindowDataset(X_test, U_test, window_size=window_size, step=step)
    
    elif dataset_type == "full":
        train_dataset = TensorDataset(X_train, U_train)
        val_dataset = TensorDataset(X_val, U_val)
        test_dataset = TensorDataset(X_test, U_test)
    
    return train_dataset, val_dataset, test_dataset, train_indices, val_indices, test_indices