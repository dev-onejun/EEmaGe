import os.path as op
import numpy as np


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
