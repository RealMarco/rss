# RSS

## Notes
### Error Debugging
1. if report an **error about CUDA version** when run **$ pip install -e rvt/libs/point-renderer**
Solution: As I have already installed several different versions of system CUDA in /usr/local/, simply reconfig **PATH and LD_LIBRARY_PATH** in the terminal using point-renderer:
```
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
# Optional to add in  ~/.bashrc or not

nvcc -V   # check
pip install -e rvt/libs/point-renderer

# install GPA-RAM
cd rvt
pip install -e . 
```

### Real-robot 
1. train
```
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
python train_real.py --exp_cfg_path configs/rvt2.yaml --mvt_cfg_path mvt/configs/rvt2.yaml --exp_cfg_opts 'epochs 15'

python train_real.py --exp_cfg_path configs/rvt2.yaml --mvt_cfg_path mvt/configs/rvt2.yaml --mvt_cfg_opts 'stage_two False' --exp_cfg_opts 'epochs 200 peract.transform_augmentation False' 

python train_real.py --exp_cfg_path configs/rvt2.yaml --mvt_cfg_path mvt/configs/rvt2.yaml --mvt_cfg_opts 'stage_two False' --exp_cfg_opts 'epochs 200 peract.transform_augmentation True' 

python train_real.py --exp_cfg_path configs/rvt2.yaml --mvt_cfg_path mvt/configs/rvt2.yaml --mvt_cfg_opts 'stage_two False' --exp_cfg_opts 'epochs 199 peract.transform_augmentation True bs 32 rvt.place_with_mean True' --train_continue

```
1. evaluation
```
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

python eval_real.py --model-folder runs  --eval-datafolder /media/marco/ubuntu_data/RVT3_real_data --tasks all --eval-episodes 25 --log-name real_world --device 0 --headless --model-name model_49.pth


~python eval_real.py --model-folder runs --eval-datafolder real_world --tasks all --eval-episodes 25 --log-name~ 
```


## Getting Started

### Install
- Tested (Recommended) Versions: Python 3.8. We used CUDA 11.1. 

- **Step 1 (Optional):**
We recommend using [conda](https://docs.conda.io/en/latest/miniconda.html) and creating a virtual environment.
```
conda create --name rvt python=3.8
conda activate rvt
```

- **Step 2:** Install PyTorch. Make sure the PyTorch version is compatible with the CUDA version. One recommended version compatible with CUDA 11.1 and PyTorch3D can be installed with the following command. More instructions to install PyTorch can be found [here](https://pytorch.org/).
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

Recently, we noticed an issue  while using conda to install PyTorch. More details can be found [here](https://github.com/pytorch/pytorch/issues/123097). If you face the same issue, you can use the following command to install PyTorch using pip.
```
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --index-url https://download.pytorch.org/whl/cu113
```

- **Step 3:** Install PyTorch3D. 

You can skip this step if you only want to use RVT-2 as it uses our custom Point-Renderer for rendering. PyTorch3D is required for RVT.

One recommended version that is compatible with the rest of the library can be installed as follows. Note that this might take some time. For more instructions visit [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).
```
curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
tar xzf 1.10.0.tar.gz
export CUB_HOME=$(pwd)/cub-1.10.0
pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'
```

- **Step 4:** Install CoppeliaSim. PyRep requires version **4.1** of CoppeliaSim. Download and unzip CoppeliaSim: 
- [Ubuntu 16.04](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Player_V4_1_0_Ubuntu16_04.tar.xz)
- [Ubuntu 18.04](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Player_V4_1_0_Ubuntu18_04.tar.xz)
- [Ubuntu 20.04](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Player_V4_1_0_Ubuntu20_04.tar.xz)

Once you have downloaded CoppeliaSim, add the following to your *~/.bashrc* file. (__NOTE__: the 'EDIT ME' in the first line)

```
export COPPELIASIM_ROOT=<EDIT ME>/PATH/TO/COPPELIASIM/INSTALL/DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export DISPLAY=:1.0
```
Remember to source your .bashrc (`source ~/.bashrc`) or  .zshrc (`source ~/.zshrc`) after this.

- **Step 5:** Clone the repository with the submodules using the following command.

```
git clone --recurse-submodules git@github.com:NVlabs/RVT.git && cd RVT && git submodule update --init
```

Now, locally install the repository. You can either `pip install -e '.[xformers]'` to install the library with [xformers](https://github.com/facebookresearch/xformers) or `pip install -e .` to install without it. We recommend using the former as improves speed. However, sometimes the installation might fail due to the xformers dependency. In that case, you can install the library without xformers. The performance difference between the two is minimal but speed could be slower without xformers.
```
pip install -e '.[xformers]' 
```

Install, required libraries for PyRep, RLBench, YARR, PerAct Colab, and Point Renderer.
```
pip install -e rvt/libs/PyRep 
pip install -e rvt/libs/RLBench 
pip install -e rvt/libs/YARR 
pip install -e rvt/libs/peract_colab
pip install -e rvt/libs/point-renderer
``` 
 
- **Step 6:** Download dataset.
    - For experiments on RLBench, we use [pre-generated dataset](https://drive.google.com/drive/folders/0B2LlLwoO3nfZfkFqMEhXWkxBdjJNNndGYl9uUDQwS1pfNkNHSzFDNGwzd1NnTmlpZXR1bVE?resourcekey=0-jRw5RaXEYRLe2W6aNrNFEQ) provided by [PerAct](https://github.com/peract/peract#download). Please download and place them under `RVT/rvt/data/xxx` where `xxx` is either `train`, `test`, or `val`.  

    - Additionally, we use the same dataloader as PerAct, which is based on [YARR](https://github.com/stepjam/YARR). YARR creates a replay buffer on the fly which can increase the startup time. We provide an option to directly load the replay buffer from the disk. We recommend using the pre-generated replay buffer (98 GB) as it reduces the startup time. You can download the replay buffer for [indidual tasks](https://huggingface.co/datasets/ankgoyal/rvt/tree/main/replay). After downloading, uncompress the replay buffer(s) (for example using the command `tar -xf <task_name>.tar.xz`) and place it under `RVT/rvt/replay/replay_xxx/<task_name>` where `xxx` is either `train` or `val`. It is useful only if you want to train RVT from scratch and not needed if you want to evaluate the pre-trained model.


## Using the library

### Training 
##### Training RVT-2

To train RVT-2 on all RLBench tasks, use the following command (from folder `RVT/rvt`):
```
python train.py --exp_cfg_path configs/rvt2.yaml --mvt_cfg_path mvt/configs/rvt2.yaml --device 0,1,2,3,4,5,6,7 
```

##### Training RVT
To train RVT, use the following command (from folder `RVT/rvt`):
```
python train.py --exp_cfg_path configs/rvt.yaml --device 0,1,2,3,4,5,6,7
```
We use 8 V100 GPUs. Change the `device` flag depending on available compute.

##### More details about `train.py`
- default parameters for an `experiment` are defined [here](https://github.com/NVlabs/RVT/blob/master/rvt/config.py).
- default parameters for `rvt` are defined [here](https://github.com/NVlabs/RVT/blob/master/rvt/mvt/config.py).
- the parameters in for `experiment` and `rvt` can be overwritten by two ways:
    - specifying the path of a yaml file
    - manually overwriting using a `opts` string of format `<param1> <val1> <param2> <val2> ..`
- Manual overwriting has higher precedence over the yaml file.

```
python train.py --exp_cfg_opts <> --mvt_cfg_opts <> --exp_cfg_path <> --mvt_cfg_path <>
```

The following command overwrites the parameters for the `experiment` with the `configs/all.yaml` file. It also overwrites the `bs` parameters through the command line.
```
python train.py --exp_cfg_opts "bs 4" --exp_cfg_path configs/rvt.yaml --device 0
```

### Evaluate on RLBench
##### Evaluate RVT-2 on RLBench
Download the [pretrained RVT-2 model](https://huggingface.co/ankgoyal/rvt/tree/main/rvt2). Place the model (`model_99.pth` trained for 99 epochs or ~80K steps with batch size 192) and the config files under the folder `RVT/rvt/runs/rvt2/`. Run evaluation using (from folder `RVT/rvt`):
```
python eval.py --model-folder runs/rvt2  --eval-datafolder ./data/test --tasks all --eval-episodes 25 --log-name test/1 --device 0 --headless --model-name model_99.pth
```
##### Evaluate RVT on RLBench
Download the [pretrained RVT model](https://huggingface.co/ankgoyal/rvt/tree/main/rvt). Place the model (`model_14.pth` trained for 15 epochs or 100K steps) and the config files under the folder `runs/rvt/`. Run evaluation using (from folder `RVT/rvt`):
```
python eval.py --model-folder runs/rvt  --eval-datafolder ./data/test --tasks all --eval-episodes 25 --log-name test/1 --device 0 --headless --model-name model_14.pth
```

##### Evaluate the official PerAct model on RLBench
Download the [officially released PerAct model](https://drive.google.com/file/d/1vc_IkhxhNfEeEbiFPHxt_AsDclDNW8d5/view?usp=share_link).
Put the downloaded policy under the `runs` folder with the recommended folder layout: `runs/peract_official/seed0`.
Run the evaluation using:
```
python eval.py --eval-episodes 25 --peract_official --peract_model_dir runs/peract_official/seed0/weights/600000 --model-name QAttentionAgent_layer0.pt --headless --task all --eval-datafolder ./data/test --device 0 
```

## Gotchas
- If you face issues installing `xformers` and PyTorch3D, information in this issue might be useful https://github.com/NVlabs/RVT/issues/45.

- If you get qt plugin error like `qt.qpa.plugin: Could not load the Qt platform plugin "xcb" <somepath>/cv2/qt/plugins" even though it was found`, try uninstalling opencv-python and installing opencv-python-headless

```
pip uninstall opencv-python                                                                                         
pip install opencv-python-headless
```

- If you have CUDA 11.7, an alternate installation strategy could be to use the following command for Step 2 and Step 3. Note that this is not heavily tested.
```
# Step 2:
pip install pytorch torchvision torchaudio
# Step 3:
pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'
```

- If you are having issues running evaluation on a headless server, please refer to https://github.com/NVlabs/RVT/issues/2#issuecomment-1620704943.

- If you want to generate visualization videos, please refer to https://github.com/NVlabs/RVT/issues/5.


## Acknowledgement
We sincerely thank the authors of the following repositories for sharing their code.

- [PerAct](https://github.com/peract/peract)
- [PerAct Colab](https://github.com/peract/peract_colab/tree/master)
- [PyRep](https://github.com/stepjam/PyRep)
- [RLBench](https://github.com/stepjam/RLBench/tree/master)
- [YARR](https://github.com/stepjam/YARR)

## License
License Copyright © 2023, NVIDIA Corporation & affiliates. All rights reserved.

This work is made available under the [Nvidia Source Code License](https://github.com/NVlabs/RVT/blob/master/LICENSE).
The pretrained RVT models are released under the CC-BY-NC-SA-4.0 license.
