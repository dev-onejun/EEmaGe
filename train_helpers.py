import os.path as op
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm


def load_losses(saved_models_dir, name):
    with open(op.join(saved_models_dir, name + "_train_losses.npy"), "rb") as f:
        train_losses = list(np.load(f))
    with open(op.join(saved_models_dir, name + "_test_losses.npy"), "rb") as f:
        test_losses = list(np.load(f))

    return train_losses, test_losses


def save_losses(train_losses, test_losses, saved_models_dir, name):
    with open(op.join(saved_models_dir, name + "_train_losses.npy"), "wb") as f:
        np.save(f, train_losses)
    with open(op.join(saved_models_dir, name + "_test_losses.npy"), "wb") as f:
        np.save(f, test_losses)


def process_large_dataset(dataloader):
    for data in tqdm(dataloader):
        yield data


def get_downstream_classification_arguments():
    parser = ArgumentParser(description="EEmaGe Classification Training Script")
    parser.add_argument(
        "--cuda",
        default=1,
        type=int,
        help="Use cuda or not (default: 1, recommend: 1|0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The random seed number to get the experiment result same",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="channelnet",
        help="model selection option (base / channelnet are supported so far) (default: channelnet)",
    )
    parser.add_argument(
        "--eeg-exclusion-channel-num",
        type=int,
        default=0,
        help="The number of unrelated EEG Channels (default: 0) (Recommend: 0|17)",
    )
    parser.add_argument(
        "--pretrained-model-path",
        type=str,
        default="./saved_models/channelnet.pt",
        help="The path to the pretrained model file",
    )
    parser.add_argument(
        "--eeg-train-data",
        type=str,
        default="./datasets/perceivelab-dataset/data/eeg_55_95_std.pth",
        help="the path for the root directory of the training dataset (default: ./datasets/perceivelab-dataset/data/eeg_55_95_std.pth)",
    )
    parser.add_argument(
        "--image-data-path",
        type=str,
        default="./datasets/imagenet-dataset/train",
        help="the path for the root directory of the image dataset (default: ./datasets/imagenet-dataset/train)",
    )
    parser.add_argument(
        "--block-splits-path",
        type=str,
        default="./datasets/perceivelab-dataset/data/block_splits_by_image_all.pth",
    )
    parser.add_argument(
        "--number-workers",
        "-n",
        type=int,
        default=16,
        metavar="N",
        help="number of works to load dataset (default: 16)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 32)",
    )
    parser.add_argument(
        "--should-shuffle",
        type=bool,
        default=False,
        help="should shuffle the images of dataset (default: False)",
    )
    parser.add_argument(
        "--downstream-task",
        type=bool,
        default=False,
        help="Execute for downstream task (default: False)",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=10,
    )

    return parser.parse_args()
