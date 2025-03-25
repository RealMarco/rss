CAMERAS = ["front", "overhead"]
SCENE_BOUNDS = [-0.77,-0.77,-0.05,0.03,0.03,0.75]  # [-0.87,-0.87,-0.10,0.13,0.13,0.90]  # [x_min, y_min, z_min, x_max, y_max, z_max] - the metric volume to be voxelized
IMAGE_SIZE = 128
VOXEL_SIZES = [100]  # 100x100x100 voxels
LOW_DIM_SIZE = 4  # {left_finger_joint, right_finger_joint, gripper_open, timestep}

DATA_FOLDER = "/media/marco/ubuntu_data/RVT3_real_data2"

RLBENCH_TASKS = [
    "put_in_drawer",
    # "insert_round_hole",
    # "pour_pills",
    # "place_intravenous_bottles",
    # "open_pill_bottle",
]
EPISODE_FOLDER = "episode%d"
VARIATION_DESCRIPTIONS_PKL = "variation_descriptions.pkl"  # the pkl file that contains language goals for each demonstration

DEMO_AUGMENTATION_EVERY_N = 10  # sample n-th frame in demo
ROTATION_RESOLUTION = 5  # degree increments per axis
# settings
NUM_LATENTS = 512  # PerceiverIO latents