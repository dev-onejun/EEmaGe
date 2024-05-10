import os.path as op

from torchvision import transforms
import numpy as np
from tqdm import tqdm


def load_losses(saved_models_dir, name):
    with open(op.join(saved_models_dir, name + "_train_losses.npy"), "rb") as f:
        train_losses = list(np.load(f))
    with open(op.join(saved_models_dir, name + "_val_losses.npy"), "rb") as f:
        val_losses = list(np.load(f))

    return train_losses, val_losses


def save_losses(train_losses, val_losses, saved_models_dir, name):
    with open(op.join(saved_models_dir, name + "_train_losses.npy"), "wb") as f:
        np.save(f, train_losses)
    with open(op.join(saved_models_dir, name + "_val_losses.npy"), "wb") as f:
        np.save(f, val_losses)


def process_large_dataset(dataloader):
    for data in tqdm(dataloader):
        yield data


# Image Trasnform
transform = transforms.Compose(
    [
        transforms.Resize((299, 299)),  # 이미지 크기를 299x299로 변경
        transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환
    ]
)
