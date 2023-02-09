import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from my_training_diff.data_preparation_utils import produce_transform_fn
from torchvision.datasets import ImageFolder
import re
from torch.utils.data import Dataset
from glob import glob
from PIL import Image
import os
import re
from pathlib import Path


def glob_re(pattern, strings):
    return filter(re.compile(pattern).match, strings)


class ImageDatasetPattern(Dataset):
    def __init__(self, root_dir, data_pattern_regex, transform, img_paths=None):
        self.transform = transform
        if img_paths is None:
            self.all_imgs = list(glob_re(data_pattern_regex, os.listdir(root_dir)))
        else:
            self.all_imgs = img_paths
        self.total_imgs = len(self.all_imgs)
        self.root_dir = root_dir

    def __len__(self):
        return self.total_imgs

    def __getitem__(self, idx):
        img_path = Path(self.root_dir) / self.all_imgs[idx]
        image = Image.open(img_path)
        tensor_image = self.transform(image)
        return tensor_image


class ImageOnlyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        data_pattern_regex: str = r"*\.jpg",
        img_paths=None,
        batch_size: int = 16,
    ) -> None:
        super().__init__()
        self.data_pattern_regex = data_pattern_regex
        self.batch_size = batch_size
        self.transform_fn = produce_transform_fn()

        # is valid_file will match the data_pattern to the file path using regex
        self.image_dataset = ImageDatasetPattern(
            root_dir=root_dir,
            data_pattern_regex=data_pattern_regex,
            transform=self.transform_fn,
            img_paths=img_paths,
        )

    def train_dataloader(self):
        return DataLoader(self.image_dataset, batch_size=self.batch_size, shuffle=True)
