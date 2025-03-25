# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import os
import time
import tqdm
import random
import yaml
import argparse

from collections import defaultdict
from contextlib import redirect_stdout

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib.pyplot as plt
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

import config as exp_cfg_mod
import rvt.models.rvt_agent_real as rvt_agent
import rvt.utils.ddp_utils as ddp_utils
import rvt.mvt.config as mvt_cfg_mod

from rvt.mvt.mvt_real import MVT
from rvt.models.rvt_agent import print_eval_log, print_loss_log
from rvt.utils.get_dataset_real import get_dataset
from rvt.utils.rvt_utils import (
    TensorboardManager,
    short_name,
    get_num_feat,
    load_agent,
    # RLBENCH_TASKS,
)
# from rvt.utils.peract_utils import (
#     CAMERAS,
#     SCENE_BOUNDS,
#     IMAGE_SIZE,
#     DATA_FOLDER,
# )
from utils.real_world_const import (
    CAMERAS,
    DATA_FOLDER,
    IMAGE_SIZE,
    SCENE_BOUNDS,
    RLBENCH_TASKS
)
# from ultralytics.yolo import detect as yolo
from PIL import Image
from libs.M2T2.demo_rlbench import get_model

def get_detect_model(device):
    return get_model(device)

losses = []

def plot_history(train_history, num_epochs, cmd_args, exp_cfg):
    mean_loss = []
    # save training curves
    log_dir = get_logdir(cmd_args, exp_cfg)
    tmp_dir = os.path.join(log_dir, 'tmp')
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    for key in train_history[0]:
        plot_path = os.path.join(tmp_dir, f'train_val_{key}_seed_{num_epochs}.png')
        plt.figure()
        train_values = [summary[key] for summary in train_history]
        avg_value = np.mean(train_values)
        
        plt.plot(np.linspace(0, 1, len(train_history)), train_values, label='train')
        # plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 2])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
        
        mean_loss.append(avg_value)
        
    losses.append(mean_loss)
    print(f'Saved plots to {log_dir}')

# new train takes the dataset as input
def train(epoch, agent, dataset, training_iterations, cmd_args, exp_cfg, rank=0):
    agent.train()
    log = defaultdict(list)

    data_iter = iter(dataset)
    iter_command = range(training_iterations)
    # for i in range(training_iterations):
    #     raw_batch = next(data_iter)
    #     batch = {
    #         k: v.to(agent._device)
    #         for k, v in raw_batch.items()
    #         if type(v) == torch.Tensor
    #     }
        # for k,v in batch.items():
        #     print("k___________v")
        #     print(batch)
        # print("demo", batch["demo"], batch["demo"].shape)
        # print("reward", batch["reward"], batch["reward"].shape)
        # print("keypoint_idx", batch["keypoint_idx"], batch["keypoint_idx"].shape)
        # exit()
    train_history = []
    for iteration in tqdm.tqdm(
        iter_command, disable=(rank != 0), position=0, leave=True
    ):
        
        raw_batch = next(data_iter)
        batch = {
            k: v.to(agent._device)
            for k, v in raw_batch.items()
            if type(v) == torch.Tensor
        }
        batch["tasks"] = raw_batch["tasks"]
        batch["lang_goal"] = raw_batch["lang_goal"]
        update_args = {
            "step": iteration,
        }

        update_args.update(
            {
                "last_loss": train_history[-1] if iteration > 0 else None,
                "replay_sample": batch,
                "backprop": True,
                "reset_log": (iteration == 0),
                "eval_log": False,
            }
        )
        result = agent.update(**update_args)
        train_history.append(result)

    if rank == 0:
        log = print_loss_log(agent)
        
    # if epoch % 9 == 0:
    plot_history(train_history, epoch, cmd_args, exp_cfg)
    return log


def save_agent(agent, path, epoch):
    model = agent._network
    optimizer = agent._optimizer
    lr_sched = agent._lr_sched

    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    torch.save(
        {
            "epoch": epoch,
            "model_state": model_state,
            "optimizer_state": optimizer.state_dict(),
            "lr_sched_state": lr_sched.state_dict(),
        },
        path,
    )


def get_tasks(exp_cfg):
    parsed_tasks = exp_cfg.tasks.split(",")
    if parsed_tasks[0] == "all":
        tasks = RLBENCH_TASKS
    else:
        tasks = parsed_tasks
    return tasks


def get_logdir(cmd_args, exp_cfg):
    log_dir = os.path.join(cmd_args.log_dir, exp_cfg.exp_id)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def dump_log(exp_cfg, mvt_cfg, cmd_args, log_dir):
    with open(f"{log_dir}/exp_cfg.yaml", "w") as yaml_file:
        with redirect_stdout(yaml_file):
            print(exp_cfg.dump())

    with open(f"{log_dir}/mvt_cfg.yaml", "w") as yaml_file:
        with redirect_stdout(yaml_file):
            print(mvt_cfg.dump())

    args = cmd_args.__dict__
    with open(f"{log_dir}/args.yaml", "w") as yaml_file:
        yaml.dump(args, yaml_file)


def experiment(rank, cmd_args, devices, port):
    """experiment.

    :param rank:
    :param cmd_args:
    :param devices: list or int. if list, we use ddp else not
    """
    device = devices[rank]
    device = f"cuda:{device}"
    ddp = len(devices) > 1
    ddp_utils.setup(rank, world_size=len(devices), port=port)

    exp_cfg = exp_cfg_mod.get_cfg_defaults()
    if cmd_args.exp_cfg_path != "":
        exp_cfg.merge_from_file(cmd_args.exp_cfg_path)
    if cmd_args.exp_cfg_opts != "":
        exp_cfg.merge_from_list(cmd_args.exp_cfg_opts.split(" "))

    if ddp:
        print(f"Running DDP on rank {rank}.")

    old_exp_cfg_peract_lr = exp_cfg.peract.lr
    old_exp_cfg_exp_id = exp_cfg.exp_id
    mamba_aug = 2
    # exp_cfg.peract.lr = 12.5
    exp_cfg.peract.lr *= len(devices) * exp_cfg.bs *mamba_aug
    if cmd_args.exp_cfg_opts != "":
        exp_cfg.exp_id += f"_{short_name(cmd_args.exp_cfg_opts)}"
    if cmd_args.mvt_cfg_opts != "":
        exp_cfg.exp_id += f"_{short_name(cmd_args.mvt_cfg_opts)}"

    if rank == 0:
        print(f"dict(exp_cfg)={dict(exp_cfg)}")
    exp_cfg.freeze()

    # Things to change
    BATCH_SIZE_TRAIN = exp_cfg.bs
    NUM_TRAIN = 10
    # to match peract, iterations per epoch
    TRAINING_ITERATIONS = int(exp_cfg.train_iter // (exp_cfg.bs * len(devices)))
    EPOCHS = exp_cfg.epochs
    TRAIN_REPLAY_STORAGE_DIR = "real_world/replay_train2"
    TEST_REPLAY_STORAGE_DIR = "real_world/replay_val2"
    log_dir = get_logdir(cmd_args, exp_cfg)
    tasks = get_tasks(exp_cfg)
    print("Training on {} tasks: {}".format(len(tasks), tasks))

    t_start = time.time()
    get_dataset_func = lambda: get_dataset(
        tasks,
        BATCH_SIZE_TRAIN,
        None,
        TRAIN_REPLAY_STORAGE_DIR,
        None,
        DATA_FOLDER,
        NUM_TRAIN,
        None,
        cmd_args.refresh_replay,
        device,
        num_workers=exp_cfg.num_workers,
        only_train=True,
        sample_distribution_mode=exp_cfg.sample_distribution_mode,
    )
    train_dataset, _ = get_dataset_func()
    t_end = time.time()
    print("Created Dataset. Time Cost: {} minutes".format((t_end - t_start) / 60.0))

    if exp_cfg.agent == "our":
        mvt_cfg = mvt_cfg_mod.get_cfg_defaults()
        if cmd_args.mvt_cfg_path != "":
            mvt_cfg.merge_from_file(cmd_args.mvt_cfg_path)
        if cmd_args.mvt_cfg_opts != "":
            mvt_cfg.merge_from_list(cmd_args.mvt_cfg_opts.split(" "))

        mvt_cfg.feat_dim = get_num_feat(exp_cfg.peract)
        mvt_cfg.freeze()

        # for maintaining backward compatibility
        assert mvt_cfg.num_rot == exp_cfg.peract.num_rotation_classes, print(
            mvt_cfg.num_rot, exp_cfg.peract.num_rotation_classes
        )

        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        # detect = yolo("/data3/sjy/code/RVT/rvt/libs/ultralytics/pt/best.pt")
        # for param in detect.parameters():
        #     param.requires_grad = False
        rvt = MVT(
            renderer_device=device,
            **mvt_cfg,
        ).to(device)
        if ddp:
            rvt = DDP(rvt, device_ids=[device], find_unused_parameters=True)
            # rvt = DDP(rvt, device_ids=[device])
        # detect_model = get_detect_model(device)
        # print(device)
        agent = rvt_agent.RVTAgent(
            # detect_model=detect_model.eval(),
            network=rvt,
            image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
            add_lang=mvt_cfg.add_lang,
            stage_two=mvt_cfg.stage_two,
            rot_ver=mvt_cfg.rot_ver,
            scene_bounds=SCENE_BOUNDS,
            cameras=CAMERAS,
            log_dir=f"{log_dir}/test_run/",
            cos_dec_max_step=EPOCHS * TRAINING_ITERATIONS,
            **exp_cfg.peract,
            **exp_cfg.rvt,
        )
        
            
        agent.build(training=True, device=device)
        if cmd_args.train_continue:
            checkpoint = torch.load('/media/marco/ubuntu_data/RVT3/rvt/runs/rvt2_E_200_PA.transform_augmentation_F_ST_F/model_last.pth')
            state_dict = checkpoint['model_state']
            # 如果模型使用 DistributedDataParallel，则 state_dict 中的 keys 需要带上 'module.' 前缀
            if isinstance(agent._network, DDP):
                # 如果模型是 DDP 模式，添加 'module.' 前缀
                new_state_dict = {}
                for k, v in state_dict.items():
                    if not k.startswith('module.'):
                        new_state_dict['module.' + k] = v  # 添加 'module.' 前缀
                    else:
                        new_state_dict[k] = v
            else:
                # 如果模型不是 DDP 模式，去掉 'module.' 前缀
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        new_state_dict[k[len('module.'):]] = v  # 去掉 'module.' 前缀
                    else:
                        new_state_dict[k] = v
            
            agent._network.load_state_dict(new_state_dict)
            agent._optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = checkpoint['epoch']
            agent._lr_sched.load_state_dict(checkpoint['lr_sched_state'])
            agent.train()
            # torch.cuda.empty_cache()
            
        else:
            start_epoch = 0
        
    else:
        assert False, "Incorrect agent"

    
    end_epoch = EPOCHS
    if exp_cfg.resume != "":
        agent_path = exp_cfg.resume
        print(f"Recovering model and checkpoint from {exp_cfg.resume}")
        epoch = load_agent(agent_path, agent, only_epoch=False)
        start_epoch = epoch + 1
    dist.barrier()

    if rank == 0:
        ## logging unchanged values to reproduce the same setting
        temp1 = exp_cfg.peract.lr
        temp2 = exp_cfg.exp_id
        exp_cfg.defrost()
        exp_cfg.peract.lr = old_exp_cfg_peract_lr
        exp_cfg.exp_id = old_exp_cfg_exp_id
        dump_log(exp_cfg, mvt_cfg, cmd_args, log_dir)
        exp_cfg.peract.lr = temp1
        exp_cfg.exp_id = temp2
        exp_cfg.freeze()
        tb = TensorboardManager(log_dir)

    print("Start training ...", flush=True)
    i = start_epoch
    while True:
        if i == end_epoch:
            break

        print(f"Rank [{rank}], Epoch [{i}]: Training on train dataset")
        out = train(i, agent, train_dataset, TRAINING_ITERATIONS, cmd_args, exp_cfg, rank)

        if rank == 0:
            tb.update("train", i, out)

        if cmd_args.train_continue:
            if rank == 0 and (i+1)%5==0:
                # TODO: add logic to only save some models
                save_agent(agent, f"{log_dir}/model_{i}.pth", i)
        else:
            if rank == 0 and (i+1)%5==0:
                # TODO: add logic to only save some models
                save_agent(agent, f"{log_dir}/model_{i}.pth", i)
        save_agent(agent, f"{log_dir}/model_last.pth", i)
        i += 1

    avg_loss_file = os.path.join(log_dir, 'average_loss.txt')
    with open(avg_loss_file, 'w') as f:
        for epoch, avg_loss in enumerate(losses):
            avg_loss_str = ', '.join([f'{value:.6f}' for value in avg_loss])  # 将每个值格式化为字符串
            f.write(f'Epoch {epoch + 1}: {avg_loss_str}\n')  # 保存所有指标的平均值
    print(f'Saved average loss to {avg_loss_file}')
    
    if rank == 0:
        tb.close()
        print("[Finish]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.set_defaults(entry=lambda cmd_args: parser.print_help())

    parser.add_argument("--refresh_replay", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--mvt_cfg_path", type=str, default="")
    parser.add_argument("--exp_cfg_path", type=str, default="")

    parser.add_argument("--mvt_cfg_opts", type=str, default="")
    parser.add_argument("--exp_cfg_opts", type=str, default="")

    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--with-eval", action="store_true", default=False)
    
    parser.add_argument("--train_continue", action="store_true", default=False)
    
    cmd_args = parser.parse_args()
    del (
        cmd_args.entry
    )  # hack for multi processing -- removes an argument called entry which is not picklable

    devices = cmd_args.device.split(",")
    devices = [int(x) for x in devices]

    port = (random.randint(0, 3000) % 3000) + 27000
    mp.spawn(experiment, args=(cmd_args, devices, port), nprocs=len(devices), join=True)
