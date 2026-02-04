import numpy as np
import robosuite as suite

def get_success(env, info):
    # Try info first
    if isinstance(info, dict) and "success" in info:
        return bool(info["success"])
    # Common robosuite internal success hook
    if hasattr(env, "_check_success"):
        return bool(env._check_success())
    if hasattr(env, "check_success"):
        return bool(env.check_success())
    return False

def rollout_once(seed=0, horizon=200):
    env = suite.make(
        env_name="PickPlace",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        horizon=horizon,
        control_freq=20,
        seed=seed,  # robosuite supports a seed param in the env API
    )
    obs = env.reset()

    t_success = None
    success = False

    for t in range(env.horizon):
        action = np.random.uniform(-1, 1, size=env.action_dim)
        obs, reward, done, info = env.step(action)

        s = get_success(env, info)
        if s and t_success is None:
            t_success = t
        success = success or s

        if done:
            break

    env.close()
    return {
        "success": bool(success),
        "time_to_success": int(t_success) if t_success is not None else None,
        "horizon": int(horizon),
    }

if __name__ == "__main__":
    results = [rollout_once(seed=i, horizon=200) for i in range(10)]
    sr = sum(r["success"] for r in results) / len(results)
    print("Success rate (random actions):", sr)
    print("Example:", results[0])
