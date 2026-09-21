# Leveraging Unlabeled Data via Semi-Supervised Contrastive Learning for Enhanced Monocular Depth Estimation

This repository contains the code implementation for the paper **"Leveraging Unlabeled Data via Semi-Supervised Contrastive Learning for Enhanced Monocular Depth Estimation".**

📄 **[Read the paper here (Pattern Recognition, 2026)]()**

![Workflow of Our Methods](./Overview.jpg)

## Installation

```bash
conda create -n TCDepth python=3.10 -y
conda activate TCDepth
pip install -r requirements.txt
```

## Dataset Preparation

This project uses the KITTI Eigen split and NYU Depth V2 datasets. After downloading and extracting the datasets, organize them under the `dataset/` directory as described below.

### KITTI Eigen Split

Download the following data:

1. [KITTI Raw Data](https://www.cvlibs.net/datasets/kitti/raw_data.php)
2. [KITTI Annotated Depth Maps](https://www.cvlibs.net/datasets/kitti/eval_depth_all.php)

Only the synchronized KITTI Raw sequences referenced by the Eigen split are required. A list of the required raw archives is also available from the [BTS repository](https://github.com/cleinc/bts/blob/master/utils/kitti_archives_to_download.txt).

Extract the KITTI Raw archives directly into:

```text
dataset/kitti_dataset/
```

After extracting `data_depth_annotated.zip`, merge the annotated training and validation sequence directories into `dataset/kitti_dataset/`:

```bash
cp -a /path/to/data_depth_annotated/train/. dataset/kitti_dataset/
cp -a /path/to/data_depth_annotated/val/. dataset/kitti_dataset/
```

The resulting directory structure should be:

```text
dataset/
└── kitti_dataset/
    ├── 2011_09_26/
    │   ├── calib_cam_to_cam.txt
    │   ├── calib_imu_to_velo.txt
    │   ├── calib_velo_to_cam.txt
    │   ├── 2011_09_26_drive_0001_sync/
    │   │   ├── image_02/
    │   │   │   └── data/
    │   │   │       ├── 0000000000.png
    │   │   │       └── ...
    │   │   └── image_03/
    │   │       └── data/
    │   └── ...
    ├── 2011_09_28/
    ├── 2011_09_29/
    ├── 2011_09_30/
    ├── 2011_10_03/
    ├── 2011_09_26_drive_0001_sync/
    │   └── proj_depth/
    │       └── groundtruth/
    │           ├── image_02/
    │           │   ├── 0000000005.png
    │           │   └── ...
    │           └── image_03/
    ├── 2011_09_26_drive_0002_sync/
    └── ...
```

The date directories contain the RGB images from KITTI Raw, while the top-level sequence directories contain the annotated ground-truth depth maps.

### NYU Depth V2

The code expects the synchronized and preprocessed NYU Depth V2 subset used by [BTS](https://github.com/cleinc/bts). It can be downloaded from:

- [Preprocessed NYU Depth V2 (`sync.zip`)](https://drive.google.com/file/d/1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP/view?usp=sharing)
- [Official NYU Depth V2 website](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)

Download `sync.zip` and extract it into:

```text
dataset/nyu_depth_v2/
```

The scene directories must be located directly inside `nyu_depth_v2`. If extraction creates an additional `sync/` directory, move its contents up one level.

The resulting directory structure should be:

```text
dataset/
└── nyu_depth_v2/
    ├── bathroom/
    │   ├── rgb_00045.jpg
    │   ├── sync_depth_00045.png
    │   ├── rgb_00046.jpg
    │   ├── sync_depth_00046.png
    │   └── ...
    ├── bedroom_0001/
    │   ├── rgb_00000.jpg
    │   ├── sync_depth_00000.png
    │   └── ...
    ├── kitchen_0001/
    ├── living_room_0001/
    ├── office_0001/
    └── ...
```

The project expects paired RGB and depth files with matching indices:

```text
rgb_00045.jpg
sync_depth_00045.png
```

The single-file `nyu_depth_v2_labeled.mat` release is not used directly by this data loader.

### Complete Dataset Layout

After preparing both datasets, the project directory should contain:

```text
TriConDepth/
├── configs/
├── data_splits/
│   ├── eigen_train_files_with_gt.txt
│   ├── eigen_test_files_with_gt.txt
│   ├── nyudepthv2_train_files_with_gt.txt
│   ├── nyudepthv2_test_files_with_gt.txt
│   ├── phase1/
│   └── phase2/
├── dataset/
│   ├── kitti_dataset/
│   │   ├── 2011_09_26/
│   │   ├── 2011_09_28/
│   │   ├── 2011_09_29/
│   │   ├── 2011_09_30/
│   │   ├── 2011_10_03/
│   │   ├── 2011_09_26_drive_0001_sync/
│   │   └── ...
│   └── nyu_depth_v2/
│       ├── bathroom/
│       ├── bedroom_0001/
│       ├── kitchen_0001/
│       └── ...
├── dataloaders/
├── models/
├── networks/
├── train.py
└── test.py
```

The dataset paths are already defined in the configuration files:

```yaml
# configs/eigen_base.yaml
dataset:
  data_path: './dataset/kitti_dataset'
  data_path_eval: './dataset/kitti_dataset'
```

```yaml
# configs/nyu_base.yaml
dataset:
  data_path: './dataset/nyu_depth_v2'
  data_path_eval: './dataset/nyu_depth_v2'
```

Each line in a split file follows this format:

```text
<RGB image path> <ground-truth depth path> <focal length>
```

For example:

```text
2011_09_26/2011_09_26_drive_0057_sync/image_02/data/0000000116.png 2011_09_26_drive_0057_sync/proj_depth/groundtruth/image_02/0000000116.png 721.5377
```

```text
/kitchen_0028b/rgb_00045.jpg /kitchen_0028b/sync_depth_00045.png 518.8579
```

The required Eigen and NYU split files are already provided under `data_splits/`; no additional split download is required.


