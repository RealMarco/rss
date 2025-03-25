CAMERAS = ["front", "overhead"]
SCENE_BOUNDS = [
    -0.3,
    -0.5,
    0.6,
    0.7,
    0.5,
    1.6,
]  # [x_min, y_min, z_min, x_max, y_max, z_max] - the metric volume to be voxelized
IMAGE_SIZE = 128
VOXEL_SIZES = [100]  # 100x100x100 voxels
LOW_DIM_SIZE = 4  # {left_finger_joint, right_finger_joint, gripper_open, timestep}

DATA_FOLDER = "/data2/sjy/data/RLbench_data/real_robot_data"

RLBENCH_TASKS = [
    "throw_away_medical_waste",
    "pull_out_syringe",
    "pour_pills",
    "place_intravenous_bottles",
    "open_pill_bottle",
]
EPISODE_FOLDER = "episode%d"
VARIATION_DESCRIPTIONS_PKL = "variation_descriptions.pkl"  # the pkl file that contains language goals for each demonstration

DEMO_AUGMENTATION_EVERY_N = 10  # sample n-th frame in demo
ROTATION_RESOLUTION = 5  # degree increments per axis
# settings
NUM_LATENTS = 512  # PerceiverIO latents