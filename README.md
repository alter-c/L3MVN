# L3MVN: Leveraging Large Language Models for Visual Target Navigation

This work is based on [L3MVN](https://github.com/ybgdgh/L3MVN).

**Author:** Dongxu Chen

**Affiliation:** BAAI

## Frontier Semantic Exploration Framework

Visual target navigation in unknown environments is a crucial problem in robotics. Despite extensive investigation of classical and learning-based approaches in the past, robots lack common-sense knowledge about household objects and layouts. Prior state-of-the-art approaches to this task rely on learning the priors during the training and typically require significant expensive resources and time for  learning. To address this, we propose a new framework for visual target navigation that leverages Large Language Models (LLM) to impart common sense for object searching. Specifically, we introduce two paradigms: (i) zero-shot and (ii) feed-forward approaches that use language to find the relevant frontier from the semantic map as a long-term goal and explore the environment efficiently. Our analysis demonstrates the notable zero-shot generalization and transfer capabilities from the use of language. Experiments on Gibson and Habitat-Matterport 3D (HM3D) demonstrate that the proposed framework significantly outperforms existing map-based methods in terms of success rate and generalization. Ablation analysis also indicates that the common-sense knowledge from the language model leads to more efficient semantic exploration. Finally, we provide a real robot experiment to verify the applicability of our framework in real-world scenarios. The supplementary video and code can be accessed via the following link: https://sites.google.com/view/l3mvn.

![image-20200706200822807](img/system.png)

<!-- ## Requirements

- Ubuntu 22.04
- Python 3.8
- [habitat-lab](https://github.com/facebookresearch/habitat-lab) -->

## Installation

The code is testing with Python 3.8 on Ubuntu 24.04 with Nvidia H100.
#### 1. Prepare Environment
- Install this repo and create conda environment:
    ```
    git clone https://github.com/alter-c/L3MVN.git
    cd L3MVN; git checkout dev-h100; 
    conda create -n l3mvn python=3.8 cmake=3.14.0 -y
    ```

#### 2. Installing Dependencies
- We use specific versions of [habitat-sim](https://github.com/facebookresearch/habitat-sim) and [habitat-lab](https://github.com/facebookresearch/habitat-lab) as below:

  - Install habitat-sim:
    ```
    conda install -n l3mvn \
      habitat-sim=0.2.2 headless \
      -c conda-forge -c aihabitat -y
    ```

  - Install habitat-lab:
    ```
    cd L3MVN
    git clone https://github.com/facebookresearch/habitat-lab.git
    cd habitat-lab; git checkout tags/challenge-2022; 
    pip install -e .
    cd ..
    ```

- Install [pytorch](https://pytorch.org/) according to your system configuration. The code is testing on pytorch v2.1.0 and cudatoolkit v12.1 on H100. If you are using conda:
    ```
    conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 \
      -c pytorch -c nvidia
    ```

- Install [detectron2](https://github.com/facebookresearch/detectron2/discussions/5200) according to your system configuration. 
    ```
    pip install --extra-index-url https://miropsota.github.io/torch_packages_builder \
      detectron2==0.6+pt2.1.0cu121
    ```

- Install other requirements.
    ```
    pip install -r requirements.txt
    ```

#### 3. Installing Other 
- Download the [segmentation model](https://drive.google.com/file/d/1U0dS44DIPZ22nTjw0RfO431zV-lMPcvv/view?usp=share_link) in RedNet/model path. We download it locally and upload it with scp.
    ```
    scp -P 22 <path_to_pth>\rednet_semmap_mp3d_40.pth \
      user@hostname:<path_to_L3MVN>/L3MVN/RedNet/model
    ```


## Download Datasets
#### 1. Set Matterport Variables
- Create Matterport account and Set following variables.
    ```
    export TOKEN_ID=<FILL IN FROM YOUR ACCOUNT INFO IN MATTERPORT>
    export TOKEN_SECRET=<FILL IN FROM YOUR ACCOUNT INFO IN MATTERPORT>
    export DATA_DIR=</path/to/L3MVN/data>
    ```

#### 2. Download HM3D datasets

- Download [HM3D](https://aihabitat.org/datasets/hm3d/) dataset using download utility and [instructions](https://github.com/facebookresearch/habitat-sim/blob/089f6a41474f5470ca10222197c23693eef3a001/datasets/HM3D.md):
    ```
    python -m habitat_sim.utils.datasets_download \
      --username $TOKEN_ID \
      --password $TOKEN_SECRET \
      --uids hm3d_minival \
      --data-path $DATA_DIR &&
    python -m habitat_sim.utils.datasets_download \
      --username $TOKEN_ID \
      --password $TOKEN_SECRET \
      --uids hm3d_val \
      --data-path $DATA_DIR 
    ```

#### 3. Download Objectnav_HM3D datasets
- Download [Objectnav_HM3D](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md):
    ```
    cd data && mkdir objectgoal_hm3d && cd objectgoal_hm3d
    wget "https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip"
    unzip objectnav_hm3d_v1.zip
    mv objectnav_hm3d_v1/* ./
    ```



#### 4. Datasets folder structure
- The code requires the datasets in a `data` folder in the following format (same as habitat-lab):
    ```
    L3MVN/
      data/
        matterport_category_mappings.tsv
        object_norm_inv_perplexity.npy
        versioned_data
        scene_datasets/
            val/
            val_mini/
        objectgoal_hm3d/
            train/
            val/
            val_mini/
    ```


## Evaluation: 
- For evaluating the pre-trained model:
  ```
  python main_llm_vis.py --split val --eval 1 --auto_gpu_config 0 \
  -n 1 --num_eval_episodes 10 --load pretrained_models/llm_model.pt \
  --use_gtsem 0 --num_local_steps 10
  ```

## Problem
- Cannot run habita-sim on H100 due to the lack of Nvidia EGL library.
- [Related Solution](https://github.com/facebookresearch/habitat-lab/blob/main/TROUBLESHOOTING.md#graphics-troubleshooting-tips)