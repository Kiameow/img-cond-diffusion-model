from argparse import Namespace
from functools import partial
from glob import glob
from multiprocessing import Pool, cpu_count
import os
from typing import List, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from UPD_study.utilities.common_data import BaseDataset, GenericDataloader


def get_files(config: Namespace, train: bool = True) -> Union[List, Tuple[List, ...]]:
    """

    tran == True: Return a list of  paths of normal samples.
    train == False:  Return a list of  paths of normal samples.

    Args:
        config (Namespace): configuration object
        train (bool): True for train images, False for test images and labels
    Returns:
        images (List): List of paths of normal files.
        masks (List): (If train == True) List of paths of segmentations.

    """

    norm_paths = sorted(
        glob(os.path.join(config.datasets_dir, 'OPMED', 'hc_FLAIR', '*.png')))
    anom_paths = sorted(glob(os.path.join(config.datasets_dir,
                                        'OPMED', 'uh_FLAIR', '*.png')))

    segmentations = sorted(glob(os.path.join(config.datasets_dir, 'OPMED', 'masks', '*.png')))
    if train:
        return norm_paths[:-len(anom_paths)]
    else:
        # In  the Dataset class bellow, 12 positive samples get a completely blank
        # segmentation due to downsampling. These are effectively negative samples and
        # considered such during the evaluation. Due to that, we return 733 normal
        # samples, so that end up with effectively 745 positive and 745 negative samples
        return norm_paths[len(norm_paths)-len(anom_paths):], anom_paths, [0] * (len(norm_paths)-len(anom_paths)), [1] * len(anom_paths), segmentations


class ImageLoader:
    def __init__(self, transforms: T.Compose):
        self.transforms = transforms

    def __call__(self, file: str) -> np.ndarray:
        image = Image.open(file)
        image = self.transforms(image)
        return image.numpy()

class ResizeWithPadding:
    def __init__(self, target_size=128):
        self.target_size = target_size

    def __call__(self, image):
        width, height = image.size
        
        # 计算缩放比例
        if width > height:
            new_width = self.target_size
            new_height = int(height * (self.target_size / width))
        else:
            new_height = self.target_size
            new_width = int(width * (self.target_size / height))
        
        # 缩放图像
        resize_transform = T.Resize(
            (new_height, new_width), 
            T.InterpolationMode.BILINEAR
        )
        resized_image = resize_transform(image)
        
        # 计算填充尺寸
        pad_left = (self.target_size - new_width) // 2
        pad_top = (self.target_size - new_height) // 2
        pad_right = self.target_size - new_width - pad_left
        pad_bottom = self.target_size - new_height - pad_top
        
        # 填充图像
        pad_transform = T.Pad((pad_left, pad_top, pad_right, pad_bottom), fill=0)
        padded_image = pad_transform(resized_image)
        
        return padded_image

class NormalDataset(BaseDataset):
    """
    Dataset class for the training set
    """

    def __init__(self, files: List, config: Namespace):
        """
        Args
            files: list of image paths for healthy OPMED images
            config: Namespace() config object
        """
        super().__init__(config)

        self.files = files
        self.transforms = T.Compose([
                 ResizeWithPadding(config.image_size),
                 T.ToTensor(),
                ])

        # with Pool(cpu_count()) as pool:
        # Wrapping inside tqdm to show progress bar
        self.preload = list(tqdm(map(ImageLoader(self.transforms), files), 'Loading OPMED images', len(files)))

    def load_file(self, file) -> np.ndarray:

        image = Image.open(file)
        image = self.transforms(image)

        return image.numpy()

    def get_sample(self, idx) -> np.ndarray:
        return self.preload[idx]

    def __len__(self):
        return len(self.files)


class AnomalDataset(Dataset):
    """
    Dataset class for the evaluation set.
    """

    def __init__(self,
                 normal_paths: List,
                 anomal_paths: List,
                 labels_normal: List,
                 labels_anomal: List,
                 segmentations: List,
                 config: Namespace):
        """
        Args:
            normal_paths (List): normal paths
            anomal_paths (List): anomal paths
            labels_normal (List): normal sample labels
            labels_anomal (List): anomal sample labels
            segmentations (List): binary segmentation masks
            config: Namespace() config object

        """

        self.center = config.center
        self.segmentations = segmentations
        self.images = anomal_paths + normal_paths
        self.labels = labels_anomal + labels_normal
        self.image_size = config.image_size

        self.image_transforms = T.Compose([
                 ResizeWithPadding(config.image_size),
                 T.ToTensor(),
                ])

        self.mask_transforms = T.Compose([
                 ResizeWithPadding(config.image_size),
                 T.ToTensor(),
                ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        """
        :param idx: Index of the file to load.
        :return: The loaded image and the binary segmentation mask.
        """

        file_path = self.images[idx]
        image = Image.open(self.images[idx])

        image = self.image_transforms(image)
        # Center input
        if self.center:
            image = (image - 0.5) * 2

        # for healthy OPMED samples, create empty mask
        if self.labels[idx] == 0:
            segmentation = torch.zeros_like(image)[0].unsqueeze(0)
        else:
            segmentation = Image.open(self.segmentations[idx])
            segmentation = self.mask_transforms(segmentation)

        return image, segmentation, file_path


def get_datasets_opmed(config: Namespace,
                    train: bool) -> Tuple[Union[BaseDataset, Dataset], Union[BaseDataset, Dataset]]:
    """
    Return pytorch Dataset instances.

    Args:
        config (Namespace): Config object.
        train (bool): True for trainloaders, False for testloader with masks.
    Returns:
        train_dataset, validation_dataset  (Tuple[BaseDataset,BaseDataset]) if train == True
        big test_dataset, small test_dataset  (Tuple[Dataset,Dataset]) if train == False

    """

    if train:

        # get list of image paths
        trainfiles = get_files(config, train)

        # percentage experiment: keep a specific percentage of the train files, or a single image.
        # for seed != 10 (stadard seed), take them from the back of the list
        if config.percentage != 100:
            if config.percentage == -1:  # single img scenario
                if config.seed == 10:
                    trainfiles = [trainfiles[0]] * 500
                else:
                    trainfiles = [trainfiles[-1]] * 500

            else:
                if config.seed == 10:
                    trainfiles = trainfiles[:int(len(trainfiles) * (config.percentage / 100))]
                else:
                    trainfiles = trainfiles[-int(len(trainfiles) * (config.percentage / 100)):]
                if len(trainfiles) < config.batch_size:
                    print(
                        f'Number of train samples ({len(trainfiles)})',
                        f' lower than batch size ({config.batch_size}). Repeating trainfiles 10 times.')
                    trainfiles = trainfiles * 10

        # calculate dataset split index
        split_idx = int(len(trainfiles) * config.normal_split)

        trainset = NormalDataset(trainfiles[:split_idx], config)
        valset = NormalDataset(trainfiles[split_idx:], config)

        return trainset, valset

    elif not train:
        # get list of img and mask paths
        normal, anomal, labels_normal, labels_anomal, segmentations = get_files(config, train)

        # calculate split indices
        split_idx = int(len(normal) * config.anomal_split)
        split_idx_anomal = int(len(anomal) * config.anomal_split)

        big = AnomalDataset(normal[:split_idx],
                            anomal[:split_idx_anomal],
                            labels_normal[:split_idx],
                            labels_anomal[:split_idx_anomal],
                            segmentations[:split_idx_anomal],
                            config)

        small = AnomalDataset(normal[split_idx:],
                              anomal[split_idx_anomal:],
                              labels_normal[split_idx:],
                              labels_anomal[split_idx_anomal:],
                              segmentations[split_idx_anomal:],
                              config)

        return big, small


def get_dataloaders_opmed(config: Namespace, train: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Return pytorch Dataloader instances.

    Args:
        config (Namespace): Config object.
        train (bool): True for trainloaders, False for testloader with masks.
    Returns:
        train_dataloader, validation_dataloader  (Tuple[DataLoader,DataLoader]) if train == True
        big test_dataloader, small test_dataloader  (Tuple[DataLoader,DataLoader]) if train == False

    """
    dset1, dset2 = get_datasets_opmed(config, train)
    return GenericDataloader(dset1, config, shuffle=train or config.shuffle), \
        GenericDataloader(dset2, config, shuffle=train or config.shuffle)
