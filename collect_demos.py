"""
collect_demos.py
----------------
Collect human demonstrations via keyboard teleoperation.
Saves observations, actions, and metadata for each episode.

Usage:
    python collect_demos.py --num_demos 15 --output_dir data/no_dr
    python collect_demos.py --num_demos 15 --dr --output_dir data/with_dr
"""

import argparse
import json
import os
import time
import numpy as np
from datetime import datetime

import robosuite as suite
from robosuite.devices import Keyboard
from robosuite.wrappers import DomainRandomizationWrapper

import config


def flatten_obs(obs_dict):
    """Convert observation dictionary to flat numpy array."""
    obs_list = []
    for key in sorted(obs_dict.keys()):
        value = obs_dict[key]
        if hasattr(value, 'flatten'):
            obs_list.append(value.flatten())
    return np.concatenate(obs_list)


def check_success(env):
    """Check if task is successful."""
    if hasattr(env, "_check_success"):
        return env._check_success()
    if hasattr(env, "check_success"):
        return env.check_success()
    # For wrapped envs
    if hasattr(env, "env"):
        return check_success(env.env)
    return False


def collect_single_demo(env, device, episode_id, seed):
    """
    Collect one demonstration episode.
    
    Returns episode_data dict or None if user quit.
    """
    
    print(f"\n{'='*50}")
    print(f"EPISODE {episode_id} (seed={seed})")
    print(f"{'='*50}")
    print("Controls:")
    print("  Arrow keys    - Move in X-Y plane")
    print("  . and ;       - Move up/down (Z axis)")
    print("  o / p         - Rotate yaw")
    print("  y / h         - Rotate pitch")
    print("  e / r         - Rotate roll")
    print("  Spacebar      - Toggle gripper")
    print("  Q             - Finish this demo and SAVE it")
    print("  ESC           - Quit without saving")
    print()
    
    # Storage for this episode
    observations = []
    actions = []
    
    # Reset environment
    obs = env.reset()
    observations.append(flatten_obs(obs))
    
    # Track success
    success = False
    time_to_success = None
    
    # Run episode
    t = 0
    quit_flag = False
    save_flag = False
    
    while t < config.HORIZON:
        # Get action from device (returns dict in newer robosuite)
        state = device.get_controller_state()
        
        # Extract values
        dpos = state["dpos"]
        drotation = state["raw_drotation"]
        grasp = state["grasp"] * 2 - 1  # Convert 0/1 to -1/+1
        reset_flag = state["reset"]
        
        # Build full action
        full_action = np.concatenate([dpos, drotation, [grasp]])
        
        # Resize if needed
        if len(full_action) > env.action_dim:
            full_action = full_action[:env.action_dim]
        elif len(full_action) < env.action_dim:
            full_action = np.pad(full_action, (0, env.action_dim - len(full_action)))
        
        # Check for save/reset (Q key)
        if reset_flag:
            save_flag = True
            print("\n  Saving demo...")
            break
        
        # Step environment
        obs, reward, done, info = env.step(full_action)
        
        # Store data
        observations.append(flatten_obs(obs))
        actions.append(full_action.copy())
        
        # Check success
        if not success:
            if info.get("success", False) or check_success(env):
                success = True
                time_to_success = t
                print(f"  ✓ SUCCESS at timestep {t}! Press Q to save this demo.")
        
        # Render
        env.render()
        
        # Slow down to real-time
        time.sleep(0.05)
        
        t += 1

    
    # Only return data if user pressed Q to save
    if not save_flag and t >= config.HORIZON:
        print("  Episode timed out. Saving anyway...")
        save_flag = True
    
    if not save_flag:
        return None
    
    # Determine failure type
    failure_type = "none" if success else "timeout"
    
    print(f"  Episode complete: success={success}, steps={t}")
    
    return {
        "observations": np.array(observations[:-1]),
        "actions": np.array(actions),
        "success": success,
        "time_to_success": time_to_success,
        "failure_type": failure_type,
        "num_steps": t,
    }


def save_episode(episode_data, episode_id, seed, domain_randomization, output_dir):
    """Save episode data to disk."""
    
    # Create directories
    episodes_dir = os.path.join(output_dir, "episodes")
    os.makedirs(episodes_dir, exist_ok=True)
    
    # Save numpy data
    npz_path = os.path.join(episodes_dir, f"ep_{episode_id:03d}.npz")
    np.savez(
        npz_path,
        observations=episode_data["observations"],
        actions=episode_data["actions"],
    )
    
    # Prepare metadata
    metadata = {
        "episode_id": episode_id,
        "seed": seed,
        "domain_randomization": "full" if domain_randomization else "none",
        "success": episode_data["success"],
        "time_to_success": episode_data["time_to_success"],
        "failure_type": episode_data["failure_type"],
        "num_steps": episode_data["num_steps"],
        "episode_path": f"episodes/ep_{episode_id:03d}.npz",
        "timestamp": datetime.now().isoformat(),
    }
    
    # Append to metadata file
    metadata_path = os.path.join(output_dir, "metadata.jsonl")
    with open(metadata_path, "a") as f:
        f.write(json.dumps(metadata) + "\n")
    
    print(f"  Saved to {npz_path}")
    
    return metadata


def make_env(domain_randomization=False, seed=None):
    """Create environment with optional domain randomization."""
    
    env = suite.make(
        env_name=config.ENV_NAME,
        robots=config.ROBOT,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        horizon=config.HORIZON,
        control_freq=config.CONTROL_FREQ,
        single_object_mode=1,
    )
    
    if domain_randomization:
        env = DomainRandomizationWrapper(
            env,
            randomize_color=False,
            randomize_camera=True,
            randomize_lighting=True,
            randomize_dynamics=False,
            camera_randomization_args={
                "camera_names": ["agentview"],
                "position_perturbation_size": 0.0001,  # Very small
                "rotation_perturbation_size": 0.0001,   # Very small
            },
            lighting_randomization_args={
                "light_names": None,

                # what to randomize
                "randomize_position": True,
                "randomize_direction": True,
                "randomize_diffuse": True,     # this is the closest “intensity-like” knob

                # keep the rest off to stay stable
                "randomize_ambient": False,
                "randomize_specular": False,
                "randomize_active": False,

                # how much to randomize
                "position_perturbation_size": 0.05,
                "direction_perturbation_size": 0.05,
                "diffuse_perturbation_size": 0.10,
            },

    )
        print(f"[ENV] Created environment WITH domain randomization (seed={seed})")
    else:
        print(f"[ENV] Created environment WITHOUT domain randomization (seed={seed})")
    
    return env


def main():
    parser = argparse.ArgumentParser(description="Collect human demonstrations")
    parser.add_argument("--num_demos", type=int, default=15, 
                        help="Number of demos to collect")
    parser.add_argument("--dr", action="store_true",
                        help="Enable domain randomization")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save demos")
    parser.add_argument("--start_seed", type=int, default=0,
                        help="Starting seed")
    parser.add_argument("--start_episode", type=int, default=0,
                        help="Starting episode ID")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"DEMO COLLECTION")
    print(f"{'='*60}")
    print(f"Domain Randomization: {'ON' if args.dr else 'OFF'}")
    print(f"Number of demos: {args.num_demos}")
    print(f"Output directory: {args.output_dir}")
    print(f"Starting seed: {args.start_seed}")
    print(f"{'='*60}\n")
    
    episode_id = args.start_episode
    seed = args.start_seed
    demos_saved = 0
    
    try:
        while demos_saved < args.num_demos:
            # Wait for user to be ready
            input(f"\nPress ENTER to start demo {episode_id + 1}/{args.num_demos}...")
            
            # Create fresh environment for each demo
            env = make_env(domain_randomization=args.dr, seed=seed)

            device = Keyboard(env=env, pos_sensitivity=5.0, rot_sensitivity=5.0)
            device.start_control()
            
            # Collect one episode
            episode_data = collect_single_demo(env, device, episode_id, seed)
            
            # Close environment
            env.close()
            
            if episode_data is None or len(episode_data["actions"]) < 10:
                print("\nSkipped - too short or no data.")
                continue
            
            # Save episode
            save_episode(
                episode_data, 
                episode_id, 
                seed, 
                args.dr, 
                args.output_dir
            )
            
            demos_saved += 1
            episode_id += 1
            seed += 1
            
            print(f"\n✓ Progress: {demos_saved}/{args.num_demos} demos saved\n")
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    finally:
        print(f"\nCollection complete. Saved {demos_saved} demos to {args.output_dir}")


if __name__ == "__main__":
    main()