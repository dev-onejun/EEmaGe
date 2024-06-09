"""
EEG-Based Visual Classification Dataset

    https://github.com/perceivelab/eeg_visual_classification

    This dataset includes EEG data from 6 subjects.
    The recording protocol included 40 object classes with 50 images each, taken from the ImageNet dataset,
    giving a total of 2,000 images. Visual stimuli were presented to the users in a block-based setting,
    with images of each class shown consecutively in a single sequence. Each image was shown for 0.5 seconds.
    A 10-second black screen (during which we kept recording EEG data) was presented between class blocks.
    The collected dataset contains in total 11,964 segments (time intervals recording the response to each image);
    36 have been excluded from the expected 6×2,000 = 12,000 segments due to low recording quality or subjects not looking at the screen,
    checked by using the eye movement data. Each EEG segment contains 128 channels, recorded for 0.5 seconds at 1 kHz sampling rate,
    represented as a 128×L matrix, with L about 500 being the number of samples contained in each segment on each channel.
    The exact duration of each signal may vary, so we discarded the first 20 samples (20 ms) to reduce interference from the previous image
    and then cut the signal to a common length of 440 samples (to account for signals with L < 500).
    The dataset includes data already filtered in three frequency ranges: 14-70Hz, 5-95Hz and 55-95Hz.

Citations

    S. Palazzo, C. Spampinato, I. Kavasidis, D. Giordano, J. Schmidt, M. Shah,
    Decoding Brain Representations by Multimodal Learning of Neural Activity and Visual Features,
    IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, 2020, doi: 10.1109/TPAMI.2020.2995909

    C. Spampinato, S. Palazzo, I. Kavasidis, D. Giordano, N. Souly, M. Shah,
    Deep Learning Human Mind for Automated Visual Classification,
    International Conference on Computer Vision and Pattern Recognition, CVPR 2017
"""

import torch
from torch.utils.data import Dataset

from PIL import Image

import os, random

time_low = 20
time_high = 460


# Dataset class
class Dataset(Dataset):

    # Constructor
    def __init__(self, eeg_data_path, image_data_path, model_type, transform):
        # Load EEG signals
        loaded = torch.load(eeg_data_path)
        self.data = loaded["dataset"]

        self.images = loaded["images"]
        self.images = [
            os.path.join(image_data_path, image[:9], image + ".JPEG")
            for image in self.images
        ]

        # Compute size
        self.size = len(self.data)

        self.model_type = model_type
        self.transform = transform

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        eeg = self.data[i]["eeg"].float().t()
        eeg = eeg[time_low:time_high, :]
        eeg = eeg.t()

        image_index = int(self.data[i]["image"])
        image = self.images[image_index]
        image = Image.open(image).convert("RGB")
        image = self.transform(image)

        if self.model_type == "channelnet":
            eeg = eeg.view(1, 128, time_high - time_low)

        # Return
        return eeg, image, eeg, image


class Splitter:
    def __init__(
        self,
        dataset,
        split_path,
        split_name="train",
        shuffle=False,
        downstream_task=False,
    ):
        # Set EEG dataset
        self.dataset = dataset

        # Load split
        loaded = torch.load(split_path)
        split = loaded["splits"][0]

        if downstream_task:
            self.split_idx = split[split_name]
        else:
            # 학습에는 train set과 validation set을 사용하고, 테스트에는 test set을 사용
            if split_name == "train":
                self.split_idx = split["train"] + split["val"]
            else:
                self.split_idx = split[split_name]

        if shuffle and split_name == "train":
            random.shuffle(self.dataset.images)

        # Compute size
        self.size = len(self.split_idx)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Get sample from dataset
        eeg1, image1, eeg2, image2 = self.dataset[self.split_idx[i]]
        # Return
        return eeg1, image1, eeg2, image2


class PerceivelabClassification(Dataset):
    def __init__(self, eeg_data_path, image_data_path, model_type):
        loaded = torch.load(eeg_data_path)

        self.data = loaded["dataset"]
        self.images = loaded["images"]
        self.labels = loaded["labels"]

        self.idx_to_label = {key: value[:9] for key, value in enumerate(self.labels)}
        self.label_to_idx = {value: key for key, value in self.idx_to_label.items()}

        self.images = [
            os.path.join(image_data_path, image[:9], image + ".JPEG")
            for image in self.images
        ]

        self.size = len(self.data)
        self.model_type = model_type

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        eeg = self.data[idx]["eeg"].float().t()
        eeg = eeg[time_low:time_high, :]
        eeg = eeg.t()

        image_index = int(self.data[idx]["label"])
        label = self.label_to_idx[self.labels[image_index]]

        if self.model_type == "channelnet":
            eeg = eeg.view(1, 128, time_high - time_low)

        return eeg, label


class ClassificationSplitter:
    def __init__(
        self,
        dataset,
        split_path,
        split_name="train",
        shuffle=False,
        downstream_task=False,
    ):
        # Set EEG dataset
        self.dataset = dataset

        # Load split
        loaded = torch.load(split_path)
        split = loaded["splits"][0]

        # 학습에는 train set과 validation set을 사용하고, 테스트에는 test set을 사용
        if downstream_task:
            self.split_idx = split[split_name]
        else:
            if split_name == "train":
                self.split_idx = split["train"] + split["val"]
            else:
                self.split_idx = split[split_name]

        if shuffle and split_name == "train":
            random.shuffle(self.dataset.images)

        # Compute size
        self.size = len(self.split_idx)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Get sample from dataset
        eeg, label = self.dataset[self.split_idx[i]]

        # Return
        return eeg, label


if __name__ == "__main__":
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.train_helpers import transform

    dataset = PerceivelabClassification(
        "./perceivelab-dataset/data/eeg_55_95_std.pth",
        "./imagenet-dataset/train",
        "channelnet",
        # transform=transform,
    )

    from torch.utils import data

    loaders = {
        split: data.DataLoader(
            ClassificationSplitter(
                dataset,
                split_name=split,
                split_path="./perceivelab-dataset/data/block_splits_by_image_all.pth",
                shuffle=False,
                downstream_task=False,
            ),
            batch_size=32,
            shuffle=False,
            num_workers=8,
        )
        for split in ["train", "val", "test"]
    }

    train_loader = loaders["train"]
    test_loader = loaders["test"]

    labels: torch.Tensor = None
    for eeg, label in test_loader:
        labels = label

    for label in labels:
        label = int(label)
        label = dataset.idx_to_label[label]
        print(f"LABEL: {label}")

    print()
    for eeg, label in test_loader:
        labels = label

    for label in labels:
        label = int(label)
        label = dataset.idx_to_label[label]
        print(f"LABEL: {label}")
