"""
debug_object_state.py
---------------------
Check what object-state contains
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import robosuite as suite

for seed in [0, 1, 2]:
    env = suite.make(
        env_name='PickPlace',
        robots='Panda',
        has_renderer=False,
        use_camera_obs=False,
        single_object_mode=1,
        seed=seed,
    )
    obs = env.reset()
    
    # Find the object position key (varies by object type)
    obj_keys = [k for k in obs.keys() if '_pos' in k and 'robot' not in k and 'eef' not in k]
    
    print(f"\nSeed {seed}:")
    print(f"  Object key: {obj_keys}")
    if obj_keys:
        print(f"  Object pos: {obs[obj_keys[0]]}")
    print(f"  object-state (first 3): {obs['object-state'][:3]}")
    
    env.close()