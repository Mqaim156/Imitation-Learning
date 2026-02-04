import os

# PATHS

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ENV SETTINGS

ENV_NAME = "PickPlace"
ROBOT = "Panda"
HORIZON = 200          # Max steps per episode
CONTROL_FREQ = 20      # Control frequency in Hz

# DEMO SETTINGS

NUM_DEMOS = 25                    # Total demos to collect (split with partner)
TRAINING_SEEDS = list(range(50))  # Seeds 0-49 for training demos
TEST_SEEDS = list(range(1000, 1050))  # Seeds 1000-1049 for testing generalization

# DR SETTINGS

DR_CONFIG = {
    "randomize_color": True,       # Randomize object/robot colors
    "randomize_camera": True,      # Randomize camera position/angle
    "randomize_lighting": True,    # Randomize light intensity/direction
    "randomize_dynamics": True,    # Randomize friction, mass, damping
    "color_randomization_args": {
        "geom_names": None,        # None = randomize all geoms
        "randomize_local": True,
        "randomize_material": True,
    },
    "camera_randomization_args": {
        "camera_names": ["frontview", "agentview"],
        "position_perturbation_size": 0.01,
        "rotation_perturbation_size": 0.02,
    },
    "lighting_randomization_args": {
        "light_names": None,       # None = randomize all lights
        "intensity_perturbation_size": 0.3,
        "position_perturbation_size": 0.1,
    },
    "dynamics_randomization_args": {
        "randomize_friction": True,
        "randomize_density": True,
    },
}

# TRAINING HYPERPARAMETERS

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
HIDDEN_SIZES = [256, 256]  # Two hidden layers with 256 neurons each

# EVAL SETTINGS

EVAL_EPISODES = 50  # Number of episodes to run during evaluation