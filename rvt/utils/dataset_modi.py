# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

# initial source: https://colab.research.google.com/drive/1HAqemP4cE81SQ6QO1-N85j5bF4C0qLs0?usp=sharing
# adapted to support loading from disk for faster initialization time

# Adapted from: https://github.com/stepjam/ARM/blob/main/arm/c2farm/launch_utils.py
import os
import torch
import pickle
import logging
import numpy as np
from typing import List

import clip
import peract_colab.arm.utils as utils

from peract_colab.rlbench.utils import get_stored_demo
from yarr.utils.observation_type import ObservationElement
from yarr.replay_buffer.replay_buffer import ReplayElement, ReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from rlbench.backend.observation import Observation
from rlbench.demo import Demo

from rvt.utils.peract_utils import LOW_DIM_SIZE, IMAGE_SIZE, CAMERAS
from rvt.libs.peract.helpers.demo_loading_utils import keypoint_discovery
from rvt.libs.peract.helpers.utils import extract_obs

'''
        batch_size=BATCH_SIZE_TRAIN,
        timesteps=1,
        disk_saving=True,
        cameras=CAMERAS,
        voxel_sizes=VOXEL_SIZES,
'''
def create_replay(
    batch_size: int,
    timesteps: int,
    disk_saving: bool,
    cameras: list,
    voxel_sizes,
    replay_size=3e5,
    add_current_pos=True,
):

    trans_indicies_size = 3 * len(voxel_sizes)
    rot_and_grip_indicies_size = 3 + 1
    gripper_pose_size = 7
    ignore_collisions_size = 1
    # 表示语言特征的最大 token 序列长度，取值为 77（对应 CLIP 模型的限制）
    max_token_seq_len = 77
    lang_feat_dim = 1024
    lang_emb_dim = 512

    # low_dim_state
    observation_elements = []
    # LOW_DIM_SIZE = 4 ->{left_finger_joint, right_finger_joint, gripper_open, timestep}
    observation_elements.append(
        ObservationElement("low_dim_state", (LOW_DIM_SIZE,), np.float32)
    )

    # rgb, depth, point cloud, intrinsics, extrinsics
    for cname in cameras:
        observation_elements.append(
            ObservationElement(
                "%s_rgb" % cname,
                (
                    3,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement(
                "%s_mask" % cname,
                (
                    3,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement(
                "%s_depth" % cname,
                (
                    1,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement(
                "%s_point_cloud" % cname,
                (
                    3,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                np.float32,
            )
        )  # see pyrep/objects/vision_sensor.py on how pointclouds are extracted from depth frames
        observation_elements.append(
            ObservationElement(
                "%s_camera_extrinsics" % cname,
                (
                    4,
                    4,
                ),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement(
                "%s_camera_intrinsics" % cname,
                (
                    3,
                    3,
                ),
                np.float32,
            )
        )

    # discretized translation, discretized rotation, discrete ignore collision, 
    # 6-DoF gripper pose, and pre-trained language embeddings
    observation_elements.extend(
        [
            ReplayElement("trans_action_indicies", (trans_indicies_size,), np.int32),
            ReplayElement(
                "rot_grip_action_indicies", (rot_and_grip_indicies_size,), np.int32
            ),
            ReplayElement("ignore_collisions", (ignore_collisions_size,), np.int32),
            ReplayElement("gripper_pose", (gripper_pose_size,), np.float32),
            ReplayElement(
                "lang_goal_embs",
                (
                    max_token_seq_len,
                    lang_emb_dim,
                ),  # extracted from CLIP's language encoder
                np.float32,
            ),
            ReplayElement(
                "lang_goal", (1,), object
            ),  # language goal string for debugging and visualization
        ]
    )

    extra_replay_elements = [
        ReplayElement("demo", (), bool),
        ReplayElement("keypoint_idx", (), int),
        ReplayElement("episode_idx", (), int),
        ReplayElement("keypoint_frame", (), int),
        ReplayElement("next_keypoint_frame", (), int),
        ReplayElement("sample_frame", (), int),
    ]
    if not add_current_pos:
        replay_buffer = (
            UniformReplayBuffer(  # all tuples in the buffer have equal sample weighting
                disk_saving=disk_saving,
                batch_size=batch_size,
                timesteps=timesteps,
                replay_capacity=int(replay_size),
                action_shape=(8,),  # 3 translation + 4 rotation quaternion + 1 gripper open
                action_dtype=np.float32,
                reward_shape=(),
                reward_dtype=np.float32,
                update_horizon=1,
                observation_elements=observation_elements,
                extra_replay_elements=extra_replay_elements,
            )
        )
    else:
        replay_buffer = (
            UniformReplayBuffer(  # all tuples in the buffer have equal sample weighting
                disk_saving=disk_saving,
                batch_size=batch_size,
                timesteps=timesteps,
                replay_capacity=int(replay_size),
                action_shape=(8 + 8,),  # 3 translation + 4 rotation quaternion + 1 gripper open + 7 current position
                action_dtype=np.float32,
                reward_shape=(),
                reward_dtype=np.float32,
                update_horizon=1,
                observation_elements=observation_elements,
                extra_replay_elements=extra_replay_elements,
            )
        )
    return replay_buffer


# discretize translation, rotation, gripper open, and ignore collision actions
def _get_action(
    obs_tp1: Observation,
    obs_tm1: Observation,
    rlbench_scene_bounds: List[float],  # metric 3D bounds of the scene
    voxel_sizes: List[int],
    rotation_resolution: int,
    crop_augmentation: bool,
):
    # 提取当前帧抓手的旋转信息并对四元数进行归一化。抓手的四元数旋转信息存储在 gripper_pose 的后四个元素中。
    quat = utils.normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    # disc_rot: 使用工具函数将抓手的四元数旋转转化为离散的欧拉角度（在给定的分辨率下） ROTATION_RESOLUTION = 5 
    disc_rot = utils.quaternion_to_discrete_euler(quat, rotation_resolution)
    # 提取抓手在三维空间中的平移坐标
    attention_coordinate = obs_tp1.gripper_pose[:3]
    trans_indicies, attention_coordinates = [], []
    bounds = np.array(rlbench_scene_bounds)
    # 从上一帧的观测中提取是否忽略碰撞的标志，将其转换为整数类型
    ignore_collisions = int(obs_tm1.ignore_collisions)
    for depth, vox_size in enumerate(
        voxel_sizes
    ):  # only single voxelization-level is used in PerAct
        # 将抓手的位置转换为体素网格的索引值
        index = utils.point_to_voxel_index(obs_tp1.gripper_pose[:3], vox_size, bounds)
        trans_indicies.extend(index.tolist())
        # 计算体素大小与场景边界的比率
        res = (bounds[3:] - bounds[:3]) / vox_size
        # 计算体素索引对应的实际坐标
        attention_coordinate = bounds[:3] + res * index
        attention_coordinates.append(attention_coordinate)

    rot_and_grip_indicies = disc_rot.tolist()
    # 获取抓手是否打开的状态，转化为浮点数
    grip = float(obs_tp1.gripper_open)
    # 将抓手的开闭状态（0 或 1）添加到旋转和抓手状态的索引中
    rot_and_grip_indicies.extend([int(obs_tp1.gripper_open)])
    return (
        trans_indicies,
        rot_and_grip_indicies,
        ignore_collisions,
        np.concatenate([obs_tp1.gripper_pose, np.array([grip])]),
        attention_coordinates,
    )


# extract CLIP language features for goal string
def _clip_encode_text(clip_model, text):
    x = clip_model.token_embedding(text).type(
        clip_model.dtype
    )  # [batch_size, n_ctx, d_model]

    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)

    emb = x.clone()
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection

    return x, emb

'''
    replay,
    task,
    task_replay_storage_folder,
    d_idx,
    i,
    obs,
    demo,
    episode_keypoints,
    cameras,
    rlbench_scene_bounds,
    voxel_sizes,
    rotation_resolution,
    crop_augmentation,
    next_keypoint_idx=next_keypoint_idx,
    description=desc,
    clip_model=clip_model,
    device=device,
'''
# 该函数将提取的关键帧和对应的数据（观测、动作、奖励、终止状态等）添加到重放缓冲区
# 个人理解：第i帧的观测数据obs，对应目标是下一个关键帧，然后关键帧的target是下下个关键帧的action
def _add_keypoints_to_replay(
    replay: ReplayBuffer,
    task: str,
    task_replay_storage_folder: str,
    episode_idx: int, # 对应d_idx是第几个demo
    sample_frame: int, # 对应demo中的i,第几帧
    inital_obs: Observation,
    demo: Demo,
    episode_keypoints: List[int],
    cameras: List[str],
    rlbench_scene_bounds: List[float],
    voxel_sizes: List[int],
    rotation_resolution: int,
    crop_augmentation: bool,
    next_keypoint_idx: int,
    description: str = "",
    clip_model=None,
    device="cpu",
    add_current_pos=True, # 是否添加当前帧的位置
):
    point = 0
    prev_action = None
    obs = inital_obs
    for k in range(
        next_keypoint_idx, len(episode_keypoints)
    ):  # confused here, it seems that there are many similar samples in the replay
        keypoint = episode_keypoints[k]  # episode_keypoints[48, 64, 78, 122, 135, 172]
        obs_tp1 = demo[keypoint]
        # 获取当前关键帧的上一帧观测数据。
        obs_tm1 = demo[max(0, keypoint - 1)]
        # 获取关键帧keypoints的action数据
        (
            trans_indicies,
            rot_grip_indicies,
            ignore_collisions,
            action,
            attention_coordinates,
        ) = _get_action(
            obs_tp1,
            obs_tm1,
            rlbench_scene_bounds,
            voxel_sizes,
            rotation_resolution,
            crop_augmentation,
        )
        # 获取当前相机的参数
        # cam_paras_dict = get_camera_parameters(inital_obs, cameras)
        
        #  terminal: 判断当前关键帧是否为最后一个关键帧，是则reward为1
        terminal = k == len(episode_keypoints) - 1
        reward = float(terminal) * 1.0 if terminal else 0
        # 提取观测数据
        obs_dict = extract_obs(
            obs,
            CAMERAS,
            t=k - next_keypoint_idx,
            prev_action=prev_action,
            episode_length=25,
            add_masks=True
        )
        tokens = clip.tokenize([description]).numpy()
        token_tensor = torch.from_numpy(tokens).to(device)
        with torch.no_grad():
            lang_feats, lang_embs = _clip_encode_text(clip_model, token_tensor)
        obs_dict["lang_goal_embs"] = lang_embs[0].float().detach().cpu().numpy()

        prev_action = np.copy(action)

        if k == 0:
            keypoint_frame = -1
        else:
            keypoint_frame = episode_keypoints[k - 1]
        if add_current_pos:
            add_action = np.concatenate([obs.gripper_pose, np.array([float(obs.gripper_open)])])
            action = np.concatenate([action, add_action])
        others = {
            "demo": True,
            "keypoint_idx": k,
            "episode_idx": episode_idx,
            "keypoint_frame": keypoint_frame,
            "next_keypoint_frame": keypoint,
            "sample_frame": sample_frame,
        }
        final_obs = {
            "trans_action_indicies": trans_indicies,
            "rot_grip_action_indicies": rot_grip_indicies,
            "gripper_pose": obs_tp1.gripper_pose,
            "lang_goal": np.array([description], dtype=object),
        }

        others.update(final_obs)
        others.update(obs_dict)

        timeout = False
        replay.add(
            task,
            task_replay_storage_folder,
            action,
            reward,
            terminal,
            timeout,
            **others
        )
        
        obs = obs_tp1
        sample_frame = keypoint
        point += 1
    # print("point-------", point, "-------")
    # print("len(episode_keypoints)-------", len(episode_keypoints), "-------")
    # print("reward", reward)
    
        # final step
    obs_dict_tp1 = extract_obs(
        obs_tp1,
        CAMERAS,
        t=k + 1 - next_keypoint_idx,
        prev_action=prev_action,
        episode_length=25,
        add_masks=True
    )
    obs_dict_tp1["lang_goal_embs"] = lang_embs[0].float().detach().cpu().numpy()

    obs_dict_tp1.pop("wrist_world_to_cam", None)
    obs_dict_tp1.update(final_obs)
    replay.add_final(task, task_replay_storage_folder, **obs_dict_tp1)

'''
            replay=train_replay_buffer,
            task=task,
            task_replay_storage_folder=train_replay_storage_folder,
            start_idx=0,
            num_demos=NUM_TRAIN,
            demo_augmentation=True,
            demo_augmentation_every_n=DEMO_AUGMENTATION_EVERY_N,
            cameras=CAMERAS,
            rlbench_scene_bounds=SCENE_BOUNDS,
            voxel_sizes=VOXEL_SIZES,
            rotation_resolution=ROTATION_RESOLUTION,
            crop_augmentation=False,
            data_path=data_path_train,
            episode_folder=EPISODE_FOLDER,
            variation_desriptions_pkl=VARIATION_DESCRIPTIONS_PKL,
            clip_model=clip_model,
            device=device,

加载演示数据并将其填充到Replay Buffer中。
它循环处理演示中的每一帧，抽取关键帧，生成相应的动作和观测数据，并依次将其添加到Replay Buffer中
'''

def fill_replay(
    replay: ReplayBuffer,
    task: str,
    task_replay_storage_folder: str,
    start_idx: int,
    num_demos: int,
    demo_augmentation: bool,
    demo_augmentation_every_n: int,
    cameras: List[str],
    rlbench_scene_bounds: List[float],  # AKA: DEPTH0_BOUNDS
    voxel_sizes: List[int],
    rotation_resolution: int,
    crop_augmentation: bool,
    data_path: str,
    episode_folder: str,
    variation_desriptions_pkl: str,
    clip_model=None,
    device="cpu",
    add_current_pos=True,
):

    disk_exist = False
    # disk_saving = True
    if replay._disk_saving:
        # 存在reply文件打印
        if os.path.exists(task_replay_storage_folder):
            print(
                "[Info] Replay dataset already exists in the disk: {}".format(
                    task_replay_storage_folder
                ),
                flush=True,
            )
            disk_exist = True
        else:
            logging.info("\t saving to disk: %s", task_replay_storage_folder)
            os.makedirs(task_replay_storage_folder, exist_ok=True)
    # reply里面有buffer，存在的时候,一般是这个
    if disk_exist:
        replay.recover_from_disk(task, task_replay_storage_folder)
    else:
        print("Filling replay ...")
        # num_demos = 100
        for d_idx in range(start_idx, start_idx + num_demos):
            print("Filling demo %d" % d_idx)
            demo = get_stored_demo(data_path=data_path, index=d_idx)

            # get language goal from disk
            # "variation_descriptions.pkl"
            varation_descs_pkl_file = os.path.join(
                data_path, episode_folder % d_idx, variation_desriptions_pkl
            )
            with open(varation_descs_pkl_file, "rb") as f:
                descs = pickle.load(f)

            # extract keypoints 论文里面启发式提取关键帧， 为一个列表
            episode_keypoints = keypoint_discovery(demo) # [48, 64, 78, 122, 135, 172]
            # print("len_demo", len(demo))
            # print("episode_keypoints", episode_keypoints)
            # exit()
            next_keypoint_idx = 0
            for i in range(len(demo) - 1):  # 0-172
                # demo_augmentation=True
                if not demo_augmentation and i > 0:
                    break
                # sample 10-th frame in demo DEMO_AUGMENTATION_EVERY_N = 10
                if i % demo_augmentation_every_n != 0:  # choose only every n-th frame
                    continue

                obs = demo[i]
                # 获取当前演示的语言描述
                desc = descs[0]
                # 更新 next_keypoint_idx，跳过已经处理过的关键帧
                # next_keypoint_idx是episode_keypoints的索引
                # while 在i=episode_keypoints[next_keypoint_idx]时，next_keypoint_idx加1
                while (
                    next_keypoint_idx < len(episode_keypoints)
                    and i >= episode_keypoints[next_keypoint_idx]
                ):
                    next_keypoint_idx += 1
                # 如果所有关键帧已处理完毕，则结束处理
                if next_keypoint_idx == len(episode_keypoints):
                    break
                # 
                _add_keypoints_to_replay(
                    replay,
                    task,
                    task_replay_storage_folder,
                    d_idx,
                    i,
                    obs,
                    demo,
                    episode_keypoints,
                    cameras,
                    rlbench_scene_bounds,
                    voxel_sizes,
                    rotation_resolution,
                    crop_augmentation,
                    next_keypoint_idx=next_keypoint_idx,
                    description=desc,
                    clip_model=clip_model,
                    device=device,
                    add_current_pos=add_current_pos,
                )
        
        # save TERMINAL info in replay_info.npy
        task_idx = replay._task_index[task]
        with open(
            os.path.join(task_replay_storage_folder, "replay_info.npy"), "wb"
        ) as fp:
            np.save(
                fp,
                replay._store["terminal"][
                    replay._task_replay_start_index[
                        task_idx
                    ] : replay._task_replay_start_index[task_idx]
                    + replay._task_add_count[task_idx].value
                ],
            )

        print("Replay filled with demos.")
