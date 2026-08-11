from torch.utils.data import DataLoader, Dataset
from easydict import EasyDict
from pytorch_lightning import LightningDataModule

from .dataloader import preprocessing_transforms, DataLoadPreprocess
from .registry import DATAMODULES

class PairedInfiniteDataset(Dataset):
    def __init__(self, labeled_dataset, unlabeled_dataset):
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.labeled_len = len(labeled_dataset)
        self.unlabeled_len = len(unlabeled_dataset)
        self.length = max(self.labeled_len, self.unlabeled_len)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        labeled_sample = self.labeled_dataset[index % self.labeled_len]
        unlabeled_sample = self.unlabeled_dataset[index % self.unlabeled_len]
        return labeled_sample, unlabeled_sample
    
@DATAMODULES.register_module('kitti_eigen')
class EigenDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.phase = self.cfg.dataset.phase
        args = {
            'filenames_file_eval': 'data_splits/eigen_test_files_with_gt.txt',
            'phase': self.phase,
            'dataset': 'kitti',
            'use_right': False,
            'data_path': self.cfg.dataset.data_path,
            'data_path_eval': self.cfg.dataset.data_path,
            'gt_path': self.cfg.dataset.data_path,
            'gt_path_eval': self.cfg.dataset.data_path,
            'do_kb_crop': self.cfg.evaluation.do_kb_crop,
            'input_height': self.cfg.dataset.input_height,
            'input_width': self.cfg.dataset.input_width,
            'do_random_rotate': True,
            'degree': 1.0,
            'max_translation_x': 8
        }
        self.args = EasyDict(args)

    def setup(self, stage: str) -> None:
        if stage == 'fit' or stage is None:
            if self.phase == 1:
                self.args.filenames_file = 'data_splits/phase1/eigen/yuhua_train_files_with_gt_50%_phase1.txt'
                self.kitti_train = DataLoadPreprocess(self.args, 'train', transform=preprocessing_transforms('train'))

            elif self.phase == 2:
                self.args.filenames_file = 'data_splits/phase2/eigen/yuhua_train_files_with_gt_50%_phase2.txt'
                self.kitti_train = DataLoadPreprocess(self.args, 'train', transform=preprocessing_transforms('train'))

            elif self.phase == 3:
                self.args.filenames_file = 'data_splits/phase2/eigen/yuhua_train_files_with_gt_50%_phase2.txt'
                self.kitti_train = DataLoadPreprocess(self.args, 'train', transform=preprocessing_transforms('train'))

            elif self.phase == 4:
                args_labeled = EasyDict(self.args.copy())
                args_unlabeled = EasyDict(self.args.copy())

                args_labeled.filenames_file = 'data_splits/phase2/eigen/yuhua_train_files_with_gt_50%_phase2.txt'
                args_labeled.phase = 2
                self.dataset_labeled = DataLoadPreprocess(args_labeled, 'train', transform=preprocessing_transforms('train'))

                args_unlabeled.filenames_file = 'data_splits/phase1/eigen/yuhua_train_files_with_gt_50%_phase1.txt'
                args_unlabeled.phase = 1
                self.dataset_unlabeled = DataLoadPreprocess(args_unlabeled, 'train', transform=preprocessing_transforms('train'))

            if self.phase >= 3:
                self.kitti_val = DataLoadPreprocess(self.args, 'online_eval', transform=preprocessing_transforms('online_eval'))

        if stage == 'test' or stage is None:
            self.kitti_test = DataLoadPreprocess(self.args, 'online_eval', transform=preprocessing_transforms('online_eval'))
 
    def train_dataloader(self):
        if self.phase == 4:
            paired_dataset = PairedInfiniteDataset(self.dataset_labeled, self.dataset_unlabeled)
            return DataLoader(
                paired_dataset,
                self.cfg.training.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=self.cfg.training.num_workers,
            )
        else:
            return DataLoader(
                self.kitti_train,
                self.cfg.training.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=self.cfg.training.num_workers,
            )

    def val_dataloader(self):
        if self.phase < 3:
            return None
        return DataLoader(
            self.kitti_val,
            1,
            shuffle=False,
            pin_memory=True,
            num_workers=4
        )

    def test_dataloader(self):
        if self.phase < 3:
            return None
        return DataLoader(
            self.kitti_test,
            1,
            shuffle=False,
            pin_memory=True,
            num_workers=4
        )
