"""
bc_policy.py
------------
Behavioral Cloning policy network.

This is a simple MLP (Multi-Layer Perceptron) that takes observations
as input and outputs actions.

Architecture:
    observation → [hidden layers] → action
    
The network learns to minimize the difference between its predicted
actions and the human's demonstrated actions.
"""

import torch
import torch.nn as nn
import numpy as np


class BCPolicy(nn.Module):
    """
    Behavioral Cloning policy network.
    
    A simple feedforward neural network:
    obs → Linear → ReLU → Linear → ReLU → ... → Linear → action
    """
    
    def __init__(self, obs_dim, action_dim, hidden_sizes=[256, 256]):
        """
        Parameters
        ----------
        obs_dim : int
            Dimension of observation vector (input size)
        action_dim : int
            Dimension of action vector (output size)
        hidden_sizes : list of int
            Sizes of hidden layers
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Build network layers
        layers = []
        prev_size = obs_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Output layer (no activation - actions can be any value)
        layers.append(nn.Linear(prev_size, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        print(f"[POLICY] Created BCPolicy:")
        print(f"         Input dim: {obs_dim}")
        print(f"         Output dim: {action_dim}")
        print(f"         Hidden layers: {hidden_sizes}")
    
    def forward(self, obs):
        """
        Forward pass: observation → action
        
        Parameters
        ----------
        obs : torch.Tensor
            Batch of observations, shape (batch_size, obs_dim)
        
        Returns
        -------
        action : torch.Tensor
            Batch of predicted actions, shape (batch_size, action_dim)
        """
        return self.network(obs)
    
    def get_action(self, obs):
        """
        Get action for a single observation (for evaluation).
        
        Parameters
        ----------
        obs : np.ndarray
            Single observation vector
        
        Returns
        -------
        action : np.ndarray
            Predicted action vector
        """
        # Convert to tensor
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs)
        
        # Add batch dimension if needed
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        # Get prediction (no gradient needed)
        with torch.no_grad():
            action = self.forward(obs)
        
        # Convert back to numpy
        return action.squeeze(0).numpy()
    
    def save(self, path):
        """Save model weights to file."""
        torch.save({
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'state_dict': self.state_dict(),
        }, path)
        print(f"[POLICY] Saved to {path}")
    
    @classmethod
    def load(cls, path, hidden_sizes=[256, 256]):
        """Load model from file."""
        checkpoint = torch.load(path)
        policy = cls(
            obs_dim=checkpoint['obs_dim'],
            action_dim=checkpoint['action_dim'],
            hidden_sizes=hidden_sizes,
        )
        policy.load_state_dict(checkpoint['state_dict'])
        print(f"[POLICY] Loaded from {path}")
        return policy