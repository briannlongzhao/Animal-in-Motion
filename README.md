# Web-Scale Collection of Video Data for 4D Animal Reconstruction
[Brian Nlong Zhao](https://briannlongzhao.github.io/about/)<sup>1,2</sup>, [Jiajun Wu](https://jiajunwu.com/)<sup>1&dagger;</sup>, [Shangzhe Wu](https://elliottwu.com/)<sup>3&dagger;</sup>

<sup>1</sup>Stanford University, <sup>2</sup>University of Illinois Urbana-Champaign, <sup>3</sup>University of Cambridge

<sup>&dagger;</sup>Equal Advising

NeurIPS 2025 Datasets and Benchmarks, CVPR 2025 CV4Animals Workshop

TLDR: A data pipeline that automatically scrapes and process in-the-wild video into object-centered crops of animals, suitable for downstream tasks such as 3D/4D reconstruction, keypoint estimation, etc. This repo includes preprocessed datasets and code to process your own datasets.

![](assets/teaser.jpg)

## Preprocessed Datasets

Both datasets have the same format and file structure and consists of video data for 23 common quadruped categories.

```shell
AiM
|--horse
|  |--AmWzveUePWU_019_001
|  |  |--00000000_mask.png
|  |  |--00000000_metadata.json
|  |  |...
```

**AiM_preview**

A small curated dataset for benchmarking and visualization purpose. See details [here](https://www.kaggle.com/datasets/932f0231547d2d31829bb099159938c6bc7358988c864a2f2aaa5cfa770dafed).

**AiM_full**

Large dataset without manual filtering, consists of 29,927 videos, totaling 2,042,781 frames.
Each data sample contains per-frame occlusion and optical flow attributes for further filtering.

Download dataset:

```shell
wget https://download.cs.stanford.edu/viscam/AiM/AiM_full.zip
```

RGB frames are not included in this full dataset. To obtain original RGB video and video frames, run

```shell
python tools/restore_rgb_data.py --data_dir=/path/to/dataset
```

Note: this script downloads videos from online sources, therefore will fail if the video source is unavailable.

## Collecting Video Dataset

### Installation 

The code is tested with CUDA 12.1

Setup conda environment:

```shell
conda env create -f environment.yml
conda activate aim
```

Set Python path:

```shell
export PYTHONPATH=$(pwd)
```

Set OpenAI API key (for generating search queries and final filtering):

```shell
export OPENAI_API_KEY=your_openai_api_key
```

Initialize Git submodules:

```shell
git submodule update --init --recursive
```

Download Grounded-SAM2 checkpoint:

```shell
cd externals/Grounded_SAM2/checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
cd ../gdino_checkpoints
chmod +x download_ckpts.sh
./download_ckpts.sh
cd ../../..
```

Download ViTPose++ checkpoint:

```shell
mkdir externals/ViTPose/ckpt
cd externals/ViTPose/ckpt
wget https://download.cs.stanford.edu/viscam/AiM/ckpt/apt36k.pth
cd ../../..
```

Download Denoising-ViT checkpoint:

```shell
mkdir externals/Denoising_ViT/ckpt
cd externals/Denoising_ViT/ckpt
wget https://huggingface.co/jjiaweiyang/DVT/resolve/main/imgnet_denoised/vit_base_patch14_dinov2.lvd142m.pth
wget https://huggingface.co/jjiaweiyang/DVT/resolve/main/voc_denoised/vit_small_patch14_dinov2.lvd142m.pth
cd ../../..
```

Or manually download from [Hugging Face](https://huggingface.co/jjiaweiyang/DVT)

### Data Pipeline

#### Download/Prepare Videos

Modify `category_ids` in `categories/__init__.py` to set animal/object categories for your data.

**Download online videos:**

```shell
python scripts/download_video.py --config configs/default.yml
```

This will produce a directory of unprocessed videos downloaded from YouTube, ready for the next preprocessing step. To download and process video data for different categories, please modify `categories` field in your config file.

**To use your own videos:**

If you want to process your own videos instead of downloading from YouTube, prepare the videos in a directory with subdirectroies with category as the subdirectory name, for example:

```shell
video
|--dog
|  |--0001.mp4
|  |--0002.mp4
...
```

Then run the following command:

```shell
python scripts/prepare_video.py --config configs/default.yml --video_dir path/to/video
```

This will prepare the videos ready for the next preprocessing step.

#### Preprocessing

```shell
python scripts/preprocess_video.py --config configs/default.yml
```

This will produce a directory of pre-filtered video clips split by shot changes.

#### Run Tracking

```shell
python scripts/track_animal.py --config configs/default.yml
```

This will produce a directory of object-centric tracks of animals.

#### Postprocessing

```shell
python scripts/build_dataset.py --config configs/default.yml
```
This will produce a dataset directory with post-filtered data split into train/test splits.

#### DINO Features

Fit a PCA matrix from data:

```shell
python scripts/extract_dino_features.py --config configs/default.yml --pca_mode=fit
```

Or download a pre-fitted PCA matrix:

```shell
cd data
wget https://download.cs.stanford.edu/viscam/AiM/ckpt/fauna_pca.bin
cd ..
```

Apply PCA matrix to data and extract DINO features:

```shell
python scripts/extract_feature.py --config configs/default.yml --pca_mode=apply --dataset_dir=path/to/dataset --pca_path=path/to/pca.bin
```

Optionally denoise the DINO features:

```shell
python scripts/extract_feature.py --config configs/default.yml --pca_mode=apply --dataset_dir=path/to/dataset --pca_path=path/to/pca.bin --denoise
```

This will add DINO features to the data in dataset directory.

## 4D-Fauna: 4D Reconstruction form Video

### Installation

Follow [3DAnimals](https://github.com/3DAnimals/3DAnimals/blob/main/INSTALL.md) to install conda environment (tested with CUDA 11.3).

Initialize Git submodules:

```shell
git submodule update --init --recursive
```

Download pretrained 3D-Fauna checkpoint:

```shell
cd externals/Animals/results/fauna
bash download_pretrained_fauna.sh
cd ../../data/tets
bash download_tets.sh
cd ../../../../
mkdir -p data/tets
cp -rv externals/Animals/data/tets/*.npz data/tets
```

Set Python path:

```shell
export PYTHONPATH=$(pwd)
```

### Run Reconstruction

```shell
python scripts/reconstruct_video_fauna.py +data_dir=/path/to/dataset
```

## Acknowledgements

We borrow and use code and models from the following repositories, thanks for their work.

- [sam2](https://github.com/facebookresearch/sam2)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)
- [ViTPose](https://github.com/ViTAE-Transformer/ViTPose)
- [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT)
- [Denoising-ViT](https://github.com/Jiawei-Yang/Denoising-ViT)
- [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [video_object_processing](https://github.com/HusamJubran/video_object_processing)
- [dino-vit-features](https://github.com/ShirAmir/dino-vit-features)
- [3DAnimals](https://github.com/3DAnimals/3DAnimals)

