import robosuite as suite
from robosuite.wrappers import DomainRandomizationWrapper
import config


def make_env(
    domain_randomization: bool = False,
    seed: int = None,
    has_renderer: bool = False,
    has_offscreen_renderer: bool = True,
    use_camera_obs: bool = False,
):
    """
    Create a PickPlace environment with optional domain randomization.
    
    Parameters
    ----------
    domain_randomization : bool
        If True, wrap environment with DomainRandomizationWrapper.
        This adds visual and physical variations to each episode.
    
    seed : int, optional
        Random seed for reproducibility. Controls initial object positions,
        robot configuration, etc.
    
    has_renderer : bool
        If True, show on-screen visualization (for human demos).
    
    has_offscreen_renderer : bool
        If True, enable offscreen rendering (for saving videos).
    
    use_camera_obs : bool
        If True, include camera images in observations.
        We use False for state-based BC (simpler).
    
    Returns
    -------
    env : robosuite environment
        Ready-to-use environment instance.
    """
    
    # Base environment configuration
    env_config = {
        "env_name": config.ENV_NAME,
        "robots": config.ROBOT,
        "has_renderer": has_renderer,
        "has_offscreen_renderer": has_offscreen_renderer,
        "use_camera_obs": use_camera_obs,
        "horizon": config.HORIZON,
        "control_freq": config.CONTROL_FREQ,
        "single_object_mode": config.SINGLE_OBJECT_MODE,
    }
    
    # Add seed if provided
    if seed is not None:
        env_config["seed"] = seed
    
    # Create base environment
    env = suite.make(**env_config)
    env.visualize(vis_settings={"grippers": True, "robots": True})
    
    # Wrap with domain randomization if requested
    if domain_randomization:
        env = DomainRandomizationWrapper(
            env,
            seed=seed,
            randomize_color=config.DR_CONFIG["randomize_color"],
            randomize_camera=config.DR_CONFIG["randomize_camera"],
            randomize_lighting=config.DR_CONFIG["randomize_lighting"],
            randomize_dynamics=config.DR_CONFIG["randomize_dynamics"],
        )
        print(f"[ENV] Created environment WITH domain randomization (seed={seed})")
    else:
        print(f"[ENV] Created environment WITHOUT domain randomization (seed={seed})")
    
    return env


def get_obs_dim(env):
    """
    Get the dimension of the observation space.
    
    For state-based observations, this is the length of the flattened
    observation vector (robot joint positions, object positions, etc.)
    """
    obs = env.reset()
    
    # robosuite returns a dict of observations
    # We concatenate all numeric arrays into one vector
    obs_dim = 0
    for key, value in obs.items():
        if hasattr(value, 'shape'):
            obs_dim += value.size
    
    return obs_dim


def get_action_dim(env):
    """
    Get the dimension of the action space.
    
    For Panda robot with gripper: 7 DOF arm + 1 gripper = 8 dimensions
    """
    return env.action_dim


def flatten_obs(obs_dict):
    """
    Convert robosuite's observation dictionary to a flat numpy array.
    
    robosuite returns observations as a dictionary like:
    {
        'robot0_joint_pos': array([...]),
        'robot0_joint_vel': array([...]),
        'object_pos': array([...]),
        ...
    }
    
    We flatten this into a single vector for the neural network.
    """
    import numpy as np
    
    obs_list = []
    for key in sorted(obs_dict.keys()):  # Sort for consistency
        value = obs_dict[key]
        if hasattr(value, 'flatten'):
            obs_list.append(value.flatten())
    
    return np.concatenate(obs_list)