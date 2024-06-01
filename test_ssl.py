from copy import deepcopy
import torch
from torch import nn
from torch.utils import data
from ignite.metrics import FID, InceptionScore
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid

from datasets import Dataset, Splitter
from models import (
    EEmaGeBase,
    EEmaGeChannelNet,
    EEmaGeBaseReconstructor,
    EEmaGeChannelNetReconstructor,
)
from utils.args import get_test_ssl_arguments
from utils.train_helpers import transform, process_large_dataset

import sys, os, random

import numpy as np
import matplotlib.pyplot as plt


def matplotlib_imshow(img):
    img = img.mean(dim=0)
    img = img / 2 + 0.5
    npimg = img.cpu().numpy()

    plt.imshow(npimg)


# TODO: is 적용을 위한 생성 파라미터 찾아볼 것
def compute_matrix(model, test_loader, writer):
    matrix_fid = FID()

    matrix_fid.reset()
    model.eval()
    sample_images = None
    with torch.no_grad():
        for batch in process_large_dataset(test_loader):
            batch = (batch[0], batch[1])
            eeg_x, image_y = tuple(b.to(device, dtype=torch.float) for b in batch)
            image_out = model(eeg_x)

            matrix_fid.update((image_out, image_y))

            sample_images = image_out

    img_grid = make_grid(sample_images)
    matplotlib_imshow(img_grid)
    writer.add_image("EEmaGe", img_grid)

    return matrix_fid.compute()  # , matrix_is.compute()


def train(model, train_dataloader, test_dataloader):
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4
    )
    loss_fn = nn.MSELoss()
    writer = SummaryWriter()

    best_epoch, best_step, best_fid, best_model_weights = 0, 0, 0.0, None
    step = 0
    for epoch in range(1, args.epoch + 1):
        model.train()
        train_epoch_loss = 0.0
        for batch in process_large_dataset(train_dataloader):
            eeg, image = batch[0].to(device), batch[1].to(device)

            output = model(eeg)
            loss = loss_fn(output, image)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item()

            step += 1
            if step % 100 == 0:
                cur_fid = compute_matrix(model, test_dataloader, writer)
                print(f"FID @ step {step}: {cur_fid}")

                if cur_fid > best_fid:
                    best_epoch, best_step, best_fid = epoch, step, cur_fid
                    best_model_weights = deepcopy(model.state_dict())

                model.train()

    print(f"Best FID: {best_fid} @ epoch {best_epoch}, step {best_step}")
    torch.save(
        best_model_weights,
        os.path.join(
            saved_models_dir,
            "best_{}_{}_{}.pt".format(args.model_type, best_epoch, best_step),
        ),
    )
    model.load_state_dict(best_model_weights)
    i = 1
    model.eval()
    with torch.no_grad():
        for batch in process_large_dataset(test_dataloader):
            batch = (batch[0], batch[1])
            eeg, image = tuple(b.to(device) for b in batch)
            image_out = model(eeg)

            for image in image_out:
                pil_image = to_pil(image)
                pil_image.save(f"{generated_images_dir}/{i}.png")
                i += 1


def main():
    if args.model_type == "base":
        pretrained = EEmaGeBase(
            eeg_exclusion_channel_num=args.eeg_exclusion_channel_num
        )
        pretrained.load_state_dict(
            torch.load(os.path.join(saved_models_dir, args.model_type + ".pt"))
        )

        model = EEmaGeBaseReconstructor(
            eeg_exclusion_channel_num=args.eeg_exclusion_channel_num
        )
        model.eeg_feature_extractor = pretrained.eeg_feature_extractor
        model.encoder = pretrained.encoder
        model.image_decoder = pretrained.image_decoder

    elif args.model_type == "channelnet":
        pretrained = EEmaGeChannelNet(
            eeg_exclusion_channel_num=args.eeg_exclusion_channel_num
        )
        pretrained.load_state_dict(
            torch.load(os.path.join(saved_models_dir, args.model_type + ".pt"))
        )

        model = EEmaGeChannelNetReconstructor()
        model.eeg_feature_extractor = pretrained.eeg_feature_extractor
        model.encoder = pretrained.encoder
        model.image_decoder = pretrained.image_decoder

    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    dataset = Dataset(
        args.eeg_train_data, args.image_data_path, args.model_type, transform
    )
    loaders = {
        split: data.DataLoader(
            Splitter(
                dataset,
                split_name=split,
                split_path=args.block_splits_path,
                shuffle=args.should_shuffle,
                downstream_task=args.downstream_task,
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.number_workers,
        )
        for split in ["train", "val", "test"]
    }

    train_loader = loaders["train"]
    test_loader = loaders["test"]

    train(model, train_loader, test_loader)


root = os.path.dirname(__file__)
saved_models_dir = os.path.join(root, "saved_models")
if not os.path.exists(saved_models_dir):
    sys.exit(f"Directory {saved_models_dir} does not exist.")
generated_images_dir = os.path.join(root, "generated_images")
os.makedirs(generated_images_dir, exist_ok=True)


args = get_test_ssl_arguments()

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

torch.manual_seed(args.seed)
random.seed(args.seed)

to_pil = ToPILImage()

if __name__ == "__main__":
    main()
