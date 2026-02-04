"""
Dataset class for loading demonstrations.
Provides batches of (observation, action) pairs for BC training.
"""

import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DemoDataset(Dataset):
    """
    PyTorch Dataset for loading demonstration data.
    
    Each item is an (observation, action) pair from the demos.
    """
    
    def __init__(self, data_dir, successful_only=True):
        """
        Parameters
        ----------
        data_dir : str
            Path to directory containing metadata.jsonl and episodes/
        successful_only : bool
            If True, only load episodes where success=True.
            This filters out failed demonstrations.
        """
        self.data_dir = data_dir
        self.observations = []
        self.actions = []
        
        # Load metadata
        metadata_path = os.path.join(data_dir, "metadata.jsonl")
        episodes = []
        with open(metadata_path, "r") as f:
            for line in f:
                ep = json.loads(line.strip())
                if successful_only and not ep["success"]:
                    continue
                episodes.append(ep)
        
        print(f"[DATASET] Loading {len(episodes)} episodes from {data_dir}")
        
        # Load all episodes
        for ep in episodes:
            ep_path = os.path.join(data_dir, ep["episode_path"])
            data = np.load(ep_path)
            
            self.observations.append(data["observations"])
            self.actions.append(data["actions"])
        
        # Concatenate all episodes
        self.observations = np.concatenate(self.observations, axis=0)
        self.actions = np.concatenate(self.actions, axis=0)
        
        # Convert to torch tensors
        self.observations = torch.FloatTensor(self.observations)
        self.actions = torch.FloatTensor(self.actions)
        
        print(f"[DATASET] Loaded {len(self)} observation-action pairs")
        print(f"[DATASET] Observation shape: {self.observations.shape}")
        print(f"[DATASET] Action shape: {self.actions.shape}")
    
    def __len__(self):
        """Number of observation-action pairs."""
        return len(self.observations)
    
    def __getitem__(self, idx):
        """Get one (observation, action) pair."""
        return self.observations[idx], self.actions[idx]
    
    def get_obs_dim(self):
        """Dimension of observation vector."""
        return self.observations.shape[1]
    
    def get_action_dim(self):
        """Dimension of action vector."""
        return self.actions.shape[1]


def create_dataloader(data_dir, batch_size=64, shuffle=True, successful_only=True):
    """
    Create a DataLoader for training.
    
    Parameters
    ----------
    data_dir : str
        Path to demo directory
    batch_size : int
        Number of samples per batch
    shuffle : bool
        Whether to shuffle data each epoch
    successful_only : bool
        Whether to filter out failed demonstrations
    
    Returns
    -------
    dataloader : DataLoader
    dataset : DemoDataset
    """
    dataset = DemoDataset(data_dir, successful_only=successful_only)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, dataset


# Quick test
if __name__ == "__main__":
    import config
    
    # Test loading
    data_dir = os.path.join(config.DATA_DIR, "no_dr")
    if os.path.exists(os.path.join(data_dir, "metadata.jsonl")):
        loader, dataset = create_dataloader(data_dir, batch_size=32)
        
        # Print one batch
        for obs, act in loader:
            print(f"Batch observation shape: {obs.shape}")
            print(f"Batch action shape: {act.shape}")
            break
    else:
        print("No data found. Run collect_demos.py first.")