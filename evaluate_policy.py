"""
evaluate_policy.py
------------------
Evaluate a trained BC policy with partial success metrics.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import numpy as np
from tqdm import tqdm

import config
from env_factory import make_env, flatten_obs
from bc_policy import BCPolicy


def run_episode(env, policy, render=False):
    obs = env.reset()
    
    # Track metrics
    min_distance_to_object = float('inf')
    gripper_closed_count = 0
    total_movement = 0
    reached_object = False  # Got within 0.05m
    attempted_grasp = False  # Closed gripper when near object
    
    success = False
    time_to_success = None
    
    for t in range(config.HORIZON):
        flat_obs = flatten_obs(obs)
        action = policy.get_action(flat_obs)
        action = np.clip(action, -1, 1)
        
        # Track distance to object
        for key in obs.keys():
            if '_to_robot0_eef_pos' in key:
                dist = np.linalg.norm(obs[key])
                min_distance_to_object = min(min_distance_to_object, dist)
                
                if dist < 0.05:
                    reached_object = True
                    if action[-1] > 0:  # Gripper closing when near
                        attempted_grasp = True
        
        # Track gripper
        if action[-1] > 0:
            gripper_closed_count += 1
        
        # Track movement
        total_movement += np.abs(action[:3]).sum()
        
        obs, reward, done, info = env.step(action)
        
        # Check success
        if not success and info.get("success", False):
            success = True
            time_to_success = t
        
        if render:
            import time
            env.render()
            time.sleep(0.05)
        
        if done:
            break
    
    # Classify failure
    if success:
        failure_type = "none"
    elif not reached_object:
        failure_type = "no_reach"
    elif not attempted_grasp:
        failure_type = "no_grasp"
    else:
        failure_type = "dropped_or_missed"
    
    return {
        "success": success,
        "time_to_success": time_to_success,
        "failure_type": failure_type,
        "min_distance": min_distance_to_object,
        "reached_object": reached_object,
        "attempted_grasp": attempted_grasp,
        "gripper_closed_ratio": gripper_closed_count / (t + 1),
        "total_movement": total_movement,
        "num_steps": t + 1,
    }


def evaluate_policy(policy_path, seeds, domain_randomization=False, num_episodes=50, render=False):
    policy = BCPolicy.load(policy_path)
    
    results = []
    seeds_to_use = seeds[:num_episodes]
    
    print(f"\n{'='*60}")
    print("POLICY EVALUATION")
    print(f"{'='*60}")
    print(f"Policy: {policy_path}")
    print(f"Domain Randomization: {'ON' if domain_randomization else 'OFF'}")
    print(f"Episodes: {len(seeds_to_use)}")
    print(f"{'='*60}\n")
    
    for seed in tqdm(seeds_to_use, desc="Evaluating"):
        env = make_env(
            domain_randomization=domain_randomization,
            seed=seed,
            has_renderer=render,
        )
        
        result = run_episode(env, policy, render=render)
        result["seed"] = seed
        results.append(result)
        
        env.close()
    
    # Compute summary
    success_rate = sum(r["success"] for r in results) / len(results)
    avg_min_dist = np.mean([r["min_distance"] for r in results])
    reach_rate = sum(r["reached_object"] for r in results) / len(results)
    grasp_rate = sum(r["attempted_grasp"] for r in results) / len(results)
    avg_gripper_ratio = np.mean([r["gripper_closed_ratio"] for r in results])
    avg_movement = np.mean([r["total_movement"] for r in results])
    
    # Failure breakdown
    failure_counts = {"none": 0, "no_reach": 0, "no_grasp": 0, "dropped_or_missed": 0}
    for r in results:
        failure_counts[r["failure_type"]] += 1
    
    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Success Rate:        {success_rate*100:.1f}%")
    print(f"Reached Object:      {reach_rate*100:.1f}%")
    print(f"Attempted Grasp:     {grasp_rate*100:.1f}%")
    print(f"Avg Min Distance:    {avg_min_dist:.3f}m")
    print(f"Avg Gripper Closed:  {avg_gripper_ratio*100:.1f}%")
    print(f"Avg Total Movement:  {avg_movement:.1f}")
    print(f"\nFailure Breakdown:")
    for ftype, count in failure_counts.items():
        pct = count / len(results) * 100
        print(f"  {ftype}: {count} ({pct:.1f}%)")
    print(f"{'='*60}\n")
    
    return {
        "policy_path": policy_path,
        "domain_randomization": domain_randomization,
        "num_episodes": len(results),
        "success_rate": success_rate,
        "reach_rate": reach_rate,
        "grasp_rate": grasp_rate,
        "avg_min_distance": avg_min_dist,
        "avg_gripper_ratio": avg_gripper_ratio,
        "avg_movement": avg_movement,
        "failure_counts": failure_counts,
        "episode_results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seeds", type=str, choices=["training", "test"], required=True)
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--dr", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    if args.seeds == "training":
        seeds = config.TRAINING_SEEDS
    else:
        seeds = config.TEST_SEEDS
    
    results = evaluate_policy(
        policy_path=args.model,
        seeds=seeds,
        domain_randomization=args.dr,
        num_episodes=args.num_episodes,
        render=args.render,
    )
    
    if args.output:
        with open(args.output, "w") as f:
            output = {k: v for k, v in results.items() if k != "episode_results"}
            json.dump(output, f, indent=2)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()