"""
train_bc.py
-----------
Train a Behavioral Cloning policy on collected demonstrations.

Usage:
    python train_bc.py --data_dir data/no_dr --output models/bc_no_dr.pt
    python train_bc.py --data_dir data/with_dr --output models/bc_with_dr.pt
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import config
from dataset import create_dataloader
from bc_policy import BCPolicy


def train(data_dir, output_path, num_epochs=100, batch_size=64, lr=1e-3):
    """
    Train BC policy on demonstrations.
    
    Parameters
    ----------
    data_dir : str
        Path to demonstration data
    output_path : str
        Where to save trained model
    num_epochs : int
        Number of training epochs
    batch_size : int
        Samples per batch
    lr : float
        Learning rate
    
    Returns
    -------
    policy : BCPolicy
        Trained policy
    losses : list
        Training loss history
    """
    
    print(f"\n{'='*60}")
    print("BEHAVIORAL CLONING TRAINING")
    print(f"{'='*60}")
    print(f"Data: {data_dir}")
    print(f"Output: {output_path}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"{'='*60}\n")
    
    # Load data
    dataloader, dataset = create_dataloader(
        data_dir, 
        batch_size=batch_size,
        shuffle=True,
        successful_only=True,
    )
    
    # Create policy
    obs_dim = dataset.get_obs_dim()
    action_dim = dataset.get_action_dim()
    policy = BCPolicy(obs_dim, action_dim, hidden_sizes=config.HIDDEN_SIZES)
    
    # Loss function: Mean Squared Error
    # We want predicted actions to match human actions
    criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    # Training loop
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for obs_batch, action_batch in dataloader:
            # Forward pass
            predicted_actions = policy(obs_batch)
            
            # Compute loss
            loss = criterion(predicted_actions, action_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # Average loss for this epoch
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    # Save trained model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    policy.save(output_path)
    
    print(f"\nTraining complete! Final loss: {losses[-1]:.6f}")
    
    return policy, losses


def main():
    parser = argparse.ArgumentParser(description="Train BC policy")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to demonstration data")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save trained model")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE,
                        help="Learning rate")
    args = parser.parse_args()
    
    train(
        data_dir=args.data_dir,
        output_path=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()