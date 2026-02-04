import numpy as np
import robosuite as suite

def main():
    # Helpful: see available env names if you're unsure
    print("Available envs:", suite.ALL_ENVIRONMENTS)

    env = suite.make(
        env_name="PickPlace",     # if this errors, pick the exact name from suite.ALL_ENVIRONMENTS
        robots="Panda",
        has_renderer=True,         # on-screen viewer
        has_offscreen_renderer=False,
        use_camera_obs=False,      # start simple: low-dim observations only
        horizon=200,
        control_freq=20,
    )

    obs = env.reset()
    for t in range(env.horizon):
        action = np.random.uniform(-1, 1, size=env.action_dim)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break

    env.close()

if __name__ == "__main__":
    main()