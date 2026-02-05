"""
dataset.py
----------
Dataset class for loading demonstrations.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DemoDataset(Dataset):
    def __init__(self, data_dir, successful_only=True, filter_idle=True, idle_threshold=0.01):
        """
        Parameters
        ----------
        data_dir : str
            Path to directory containing metadata.jsonl and episodes/
        successful_only : bool
            If True, only load successful episodes
        filter_idle : bool
            If True, remove timesteps where the robot isn't moving
        idle_threshold : float
            Minimum action magnitude to count as "moving"
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
        
        total_frames = 0
        kept_frames = 0
        
        # Load all episodes
        for ep in episodes:
            ep_path = os.path.join(data_dir, ep["episode_path"])
            data = np.load(ep_path)
            
            obs = data["observations"]
            act = data["actions"]
            
            total_frames += len(act)
            
            if filter_idle:
                # Keep only frames where robot is actually moving
                # Check first 6 dims (position/rotation), ignore gripper
                movement = np.abs(act[:, :6]).sum(axis=1)
                moving_mask = movement > idle_threshold
                
                # Also keep frames near the moving frames (context)
                # This keeps some "before action" frames
                expanded_mask = moving_mask.copy()
                for i in range(len(moving_mask)):
                    if moving_mask[i]:
                        # Keep 2 frames before and after each moving frame
                        start = max(0, i - 2)
                        end = min(len(moving_mask), i + 3)
                        expanded_mask[start:end] = True
                
                obs = obs[expanded_mask]
                act = act[expanded_mask]
            
            kept_frames += len(act)
            
            self.observations.append(obs)
            self.actions.append(act)
        
        # Concatenate all episodes
        self.observations = np.concatenate(self.observations, axis=0)
        self.actions = np.concatenate(self.actions, axis=0)
        
        # Convert to torch tensors
        self.observations = torch.FloatTensor(self.observations)
        self.actions = torch.FloatTensor(self.actions)
        
        print(f"[DATASET] Kept {kept_frames}/{total_frames} frames ({100*kept_frames/total_frames:.1f}%)")
        print(f"[DATASET] Final dataset: {len(self)} observation-action pairs")
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]
    
    def get_obs_dim(self):
        return self.observations.shape[1]
    
    def get_action_dim(self):
        return self.actions.shape[1]


def create_dataloader(data_dir, batch_size=64, shuffle=True, successful_only=True):
    dataset = DemoDataset(data_dir, successful_only=successful_only)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, dataset