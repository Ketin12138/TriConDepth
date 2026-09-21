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

This project uses the **KITTI Eigen split** and **NYU Depth V2** datasets. Please download the datasets from the links below and organize them under the `dataset/` directory following the specified structure.

## KITTI Eigen Split

Download:

- [KITTI Raw Data](https://www.cvlibs.net/datasets/kitti/raw_data.php)
- [KITTI Annotated Depth Maps](https://www.cvlibs.net/datasets/kitti/eval_depth_all.php)
- [Required KITTI Raw archives for the Eigen split](https://github.com/cleinc/bts/blob/master/utils/kitti_archives_to_download.txt)

The expected directory structure is:

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
    │           └── image_03/
    └── ...
```

## NYU Depth V2

Download:

- [Preprocessed NYU Depth V2 (`sync.zip`)](https://drive.google.com/file/d/1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP/view?usp=sharing)
- [Official NYU Depth V2 website](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)

> **Note:** This project uses the synchronized preprocessed subset rather than the single-file `nyu_depth_v2_labeled.mat` release.

The expected directory structure is:

```text
dataset/
└── nyu_depth_v2/
    ├── bathroom/
    │   ├── rgb_00045.jpg
    │   ├── sync_depth_00045.png
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

## Dataset Layout

After preparation, the project should follow the structure below:

```text
TriConDepth/
├── configs/
├── data_splits/
├── dataset/
│   ├── kitti_dataset/
│   └── nyu_depth_v2/
├── dataloaders/
├── models/
├── networks/
├── train.py
└── test.py
```

## Training

All models in our experiments are trained using a **ResNet-101** encoder.

Although TriConDepth is described as a three-stage framework in the paper, the released training code is organized into **four phases** for implementation convenience. In particular, the supervised warm-up procedure is implemented as an independent training phase before the final semi-supervised optimization.

The four training phases are:

1. **Phase 1:** Image-level contrastive learning.
2. **Phase 2:** Window-level contrastive learning.
3. **Phase 3:** Supervised warm-up using the available labeled data.
4. **Phase 4:** Final semi-supervised training using both labeled and unlabeled data.

The four phases should be trained sequentially.

### Select the Labeled Data Ratio

Before training, first determine the proportion of labeled data to be used. The corresponding split files are provided under `data_splits/`.

Below we use **KITTI with 5% labeled data (1/20 labels)** as an example.

Open:

```text
dataloaders/eigen_datamodule.py
```

and set the split files in the `setup()` function as follows:

```python
def setup(self, stage: str) -> None:
    if stage == 'fit' or stage is None:
        if self.phase == 1:
            self.args.filenames_file = (
                'data_splits/phase1/eigen/'
                'yuhua_train_files_with_gt_5%_phase1.txt'
            )
            self.kitti_train = DataLoadPreprocess(
                self.args,
                'train',
                transform=preprocessing_transforms('train')
            )

        elif self.phase == 2:
            self.args.filenames_file = (
                'data_splits/phase2/eigen/'
                'yuhua_train_files_with_gt_5%_phase2.txt'
            )
            self.kitti_train = DataLoadPreprocess(
                self.args,
                'train',
                transform=preprocessing_transforms('train')
            )

        elif self.phase == 3:
            self.args.filenames_file = (
                'data_splits/phase2/eigen/'
                'yuhua_train_files_with_gt_5%_phase2.txt'
            )
            self.kitti_train = DataLoadPreprocess(
                self.args,
                'train',
                transform=preprocessing_transforms('train')
            )

        elif self.phase == 4:
            args_labeled = EasyDict(self.args.copy())
            args_unlabeled = EasyDict(self.args.copy())

            args_labeled.filenames_file = (
                'data_splits/phase2/eigen/'
                'yuhua_train_files_with_gt_5%_phase2.txt'
            )
            args_labeled.phase = 2
            self.dataset_labeled = DataLoadPreprocess(
                args_labeled,
                'train',
                transform=preprocessing_transforms('train')
            )

            args_unlabeled.filenames_file = (
                'data_splits/phase1/eigen/'
                'yuhua_train_files_with_gt_50%_phase1.txt'
            )
            args_unlabeled.phase = 1
            self.dataset_unlabeled = DataLoadPreprocess(
                args_unlabeled,
                'train',
                transform=preprocessing_transforms('train')
            )

        if self.phase >= 3:
            self.kitti_val = DataLoadPreprocess(
                self.args,
                'online_eval',
                transform=preprocessing_transforms('online_eval')
            )

    if stage == 'test' or stage is None:
        self.kitti_test = DataLoadPreprocess(
            self.args,
            'online_eval',
            transform=preprocessing_transforms('online_eval')
        )
```

The split files used by each phase are therefore:

```text
Phase 1:
data_splits/phase1/eigen/yuhua_train_files_with_gt_5%_phase1.txt

Phase 2:
data_splits/phase2/eigen/yuhua_train_files_with_gt_5%_phase2.txt

Phase 3:
data_splits/phase2/eigen/yuhua_train_files_with_gt_5%_phase2.txt

Phase 4 (labeled):
data_splits/phase2/eigen/yuhua_train_files_with_gt_5%_phase2.txt

Phase 4 (unlabeled):
data_splits/phase1/eigen/yuhua_train_files_with_gt_50%_phase1.txt
```

For other labeled-data settings, replace the corresponding split files with those provided in `data_splits/`.

### Training Procedure

Training should be performed sequentially from **Phase 1 to Phase 4**:

```text
Phase 1
   ↓
Phase 2
   ↓
Phase 3 (Warm-up)
   ↓
Phase 4 (Semi-supervised Training)
```

The checkpoint obtained from each phase is used to initialize the subsequent phase.




