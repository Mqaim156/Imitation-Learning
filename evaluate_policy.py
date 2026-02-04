"""
evaluate_policy.py
------------------
Evaluate a trained BC policy on test episodes.
Measures success rate, time-to-success, and failure modes.

Usage:
    python evaluate_policy.py --model models/bc_no_dr.pt --seeds training
    python evaluate_policy.py --model models/bc_no_dr.pt --seeds test
"""

import argparse
import json
import os
import numpy as np
from tqdm import tqdm

import config
from env_factory import make_env, flatten_obs
from bc_policy import BCPolicy


def evaluate_policy(
    policy_path,
    seeds,
    domain_randomization=False,
    num_episodes=50,
    render=False,
):
    """
    Evaluate a policy on a set of seeds.
    
    Parameters
    ----------
    policy_path : str
        Path to trained policy .pt file
    seeds : list of int
        Seeds to evaluate on
    domain_randomization : bool
        Whether to enable DR during evaluation
    num_episodes : int
        Number of episodes to run
    render : bool
        Whether to show visualization
    
    Returns
    -------
    results : dict
        Contains success_rate, avg_time_to_success, failure_counts, episode_results
    """
    
    # Load policy
    policy = BCPolicy.load(policy_path)
    
    # Results storage
    episode_results = []
    successes = 0
    times_to_success = []
    failure_counts = {
        "no_grasp": 0,
        "dropped": 0,
        "wrong_place": 0,
        "timeout": 0,
        "none": 0,  # Successful episodes
    }
    
    # Evaluate on each seed
    seeds_to_use = seeds[:num_episodes]
    
    print(f"\n{'='*60}")
    print("POLICY EVALUATION")
    print(f"{'='*60}")
    print(f"Policy: {policy_path}")
    print(f"Domain Randomization: {'ON' if domain_randomization else 'OFF'}")
    print(f"Episodes: {num_episodes}")
    print(f"Seeds: {seeds_to_use[0]} to {seeds_to_use[-1]}")
    print(f"{'='*60}\n")
    
    for seed in tqdm(seeds_to_use, desc="Evaluating"):
        # Create environment
        env = make_env(
            domain_randomization=domain_randomization,
            seed=seed,
            has_renderer=render,
            has_offscreen_renderer=False,
        )
        
        # Run episode
        result = run_episode(env, policy, render=render)
        result["seed"] = seed
        episode_results.append(result)
        
        # Track metrics
        if result["success"]:
            successes += 1
            times_to_success.append(result["time_to_success"])
            failure_counts["none"] += 1
        else:
            failure_counts[result["failure_type"]] += 1
        
        env.close()
    
    # Compute summary statistics
    success_rate = successes / len(seeds_to_use)
    avg_time = np.mean(times_to_success) if times_to_success else None
    
    results = {
        "policy_path": policy_path,
        "domain_randomization": domain_randomization,
        "num_episodes": len(seeds_to_use),
        "success_rate": success_rate,
        "avg_time_to_success": avg_time,
        "failure_counts": failure_counts,
        "episode_results": episode_results,
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Success Rate: {success_rate*100:.1f}%")
    if avg_time:
        print(f"Avg Time to Success: {avg_time:.1f} steps")
    print(f"\nFailure Breakdown:")
    for failure_type, count in failure_counts.items():
        if failure_type != "none":
            pct = count / len(seeds_to_use) * 100
            print(f"  {failure_type}: {count} ({pct:.1f}%)")
    print(f"{'='*60}\n")
    
    return results


def run_episode(env, policy, render=False):
    """
    Run a single evaluation episode.
    
    Returns
    -------
    result : dict
        success, time_to_success, failure_type, trajectory_info
    """
    
    obs = env.reset()
    
    success = False
    time_to_success = None
    
    # For failure analysis
    gripper_closed_steps = 0
    object_grasped = False
    max_object_height = 0
    
    for t in range(config.HORIZON):
        # Get action from policy
        obs_flat = flatten_obs(obs)
        action = policy.get_action(obs_flat)
        
        # Clip action to valid range
        action = np.clip(action, -1, 1)
        
        # Step environment
        obs, reward, done, info = env.step(action)
        
        # Track gripper state (last action dimension is gripper)
        if action[-1] > 0:  # Gripper closing
            gripper_closed_steps += 1
        
        # Track object height for failure analysis
        if "object_pos" in obs:
            obj_height = obs["object_pos"][2]  # z-coordinate
            max_object_height = max(max_object_height, obj_height)
            if obj_height > 0.1 and gripper_closed_steps > 5:
                object_grasped = True
        
        # Check success
        if not success:
            if info.get("success", False) or check_success(env):
                success = True
                time_to_success = t
        
        if render:
            env.render()
        
        if done:
            break
    
    # Classify failure
    if success:
        failure_type = "none"
    else:
        failure_type = classify_failure(
            gripper_closed_steps, 
            object_grasped, 
            max_object_height,
            t
        )
    
    return {
        "success": success,
        "time_to_success": time_to_success,
        "failure_type": failure_type,
        "num_steps": t + 1,
        "gripper_closed_steps": gripper_closed_steps,
        "object_grasped": object_grasped,
        "max_object_height": max_object_height,
    }


def check_success(env):
    """Check task success."""
    if hasattr(env, "_check_success"):
        return env._check_success()
    if hasattr(env, "check_success"):
        return env.check_success()
    # For wrapped envs
    if hasattr(env, "env"):
        return check_success(env.env)
    return False


def classify_failure(gripper_closed_steps, object_grasped, max_object_height, num_steps):
    """
    Classify why the episode failed.
    
    Categories:
    - no_grasp: Never successfully grasped the object
    - dropped: Grasped but dropped before reaching bin
    - wrong_place: Placed object but not in bin
    - timeout: Ran out of time
    """
    
    if gripper_closed_steps < 10:
        # Never really tried to grasp
        return "no_grasp"
    
    if not object_grasped:
        # Tried but failed to grasp
        return "no_grasp"
    
    if object_grasped and max_object_height < 0.15:
        # Grasped but dropped quickly
        return "dropped"
    
    if object_grasped and max_object_height >= 0.15:
        # Got object up but didn't place correctly
        return "wrong_place"
    
    return "timeout"


def main():
    parser = argparse.ArgumentParser(description="Evaluate BC policy")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model")
    parser.add_argument("--seeds", type=str, choices=["training", "test"], 
                        required=True,
                        help="Which seed set to evaluate on")
    parser.add_argument("--num_episodes", type=int, default=50,
                        help="Number of episodes")
    parser.add_argument("--dr", action="store_true",
                        help="Enable domain randomization during evaluation")
    parser.add_argument("--render", action="store_true",
                        help="Show visualization")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()
    
    # Select seeds
    if args.seeds == "training":
        seeds = config.TRAINING_SEEDS
    else:
        seeds = config.TEST_SEEDS
    
    # Run evaluation
    results = evaluate_policy(
        policy_path=args.model,
        seeds=seeds,
        domain_randomization=args.dr,
        num_episodes=args.num_episodes,
        render=args.render,
    )
    
    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            # Convert numpy types for JSON serialization
            results_json = {
                k: (v.tolist() if hasattr(v, 'tolist') else v)
                for k, v in results.items()
                if k != "episode_results"  # Skip detailed results for summary
            }
            json.dump(results_json, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()