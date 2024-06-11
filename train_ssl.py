import torch
from torch import nn
from torch.utils import data
from torch.utils.tensorboard.writer import SummaryWriter

from datasets import Dataset, Splitter
from models import EEmaGeBase, EEmaGeChannelNet
from utils.train_helpers import (
    load_losses,
    save_losses,
    transform,
    process_large_dataset,
)
from utils.args import get_train_ssl_arguments

import os, random
from copy import deepcopy
from datetime import datetime


def _eval_loss(model, val_loader, eeg_criterion, image_criterion):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for data in process_large_dataset(val_loader):
            eeg_x, image_x, eeg_y, image_y = data
            eeg_x, image_x, eeg_y, image_y = (
                eeg_x.to(device, dtype=torch.float),
                image_x.to(device, dtype=torch.float),
                eeg_y.to(device, dtype=torch.float),
                image_y.to(device, dtype=torch.float),
            )

            eeg_out, image_out = model(eeg_x, image_x)
            if args.model_type == "channelnet":
                eeg_out = eeg_out.view(-1, 1, eeg_out.size(1), eeg_out.size(2))

            eeg_loss = eeg_criterion(eeg_out, eeg_y)
            image_loss = image_criterion(image_out, image_y)
            loss = 0.5 * eeg_loss + 0.5 * image_loss

            running_loss += loss.item()

        epoch_val_loss = running_loss / len(val_loader)

    return epoch_val_loss


def _train_loss(model, train_loader, optimizer, eeg_criterion, image_criterion):
    model.train()

    total_loss = 0.0
    # epoch_train_acc = 0.0

    for data in process_large_dataset(train_loader):
        eeg_x, image_x, eeg_y, image_y = data
        eeg_x, image_x, eeg_y, image_y = (
            eeg_x.to(device, dtype=torch.float),
            image_x.to(device, dtype=torch.float),
            eeg_y.to(device, dtype=torch.float),
            image_y.to(device, dtype=torch.float),
        )

        optimizer.zero_grad()

        eeg_out, image_out = model(eeg_x, image_x)
        if args.model_type == "channelnet":
            eeg_out = eeg_out.view(-1, 1, eeg_out.size(1), eeg_out.size(2))

        eeg_loss = eeg_criterion(eeg_out, eeg_y)
        image_loss = image_criterion(image_out, image_y)
        loss = 0.5 * eeg_loss + 0.5 * image_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    epoch_train_loss = total_loss / len(train_loader)

    return epoch_train_loss


def mean_absolute_average_error(y_true, y_pred):
    loss = torch.abs(
        (y_true - y_pred) / torch.maximum(torch.mean(y_true), torch.tensor(1e-7))
    )
    loss = torch.mean(loss)
    loss = loss.to(device, dtype=torch.float)

    return loss


def _train_val_loop(model, train_loader, val_loader, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    eeg_criterion = nn.MSELoss()
    image_criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=0.9
    )

    train_losses, val_losses = [], []

    best_epoch, best_loss, best_model_weights = 0, float("inf"), None
    for epoch in range(1, epochs + 1):
        train_loss = _train_loss(
            model, train_loader, optimizer, eeg_criterion, image_criterion
        )

        train_losses.append(train_loss)

        val_loss = _eval_loss(model, val_loader, eeg_criterion, image_criterion)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch}, \t Train loss {train_loss: .4f}, \t Val loss {val_loss: .4f}"
        )

        scheduler.step()

        if epoch % args.step_size == 0:
            torch.save(
                model.state_dict(),
                os.path.join(
                    root,
                    "saved_models",
                    args.model_type + "_epoch{}_{}.pt".format(epoch, datetime.now()),
                ),
            )
            save_losses(
                train_losses,
                val_losses,
                saved_models_dir,
                args.save_losses + "_epoch{}".format(epoch),
            )

        if val_loss < best_loss:
            best_epoch, best_loss = epoch, val_loss
            best_model_weights = deepcopy(model.state_dict())

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        writer.flush()

    torch.save(
        model.state_dict(),
        os.path.join(
            root,
            "saved_models",
            "final_{}_{}.pt".format(args.model_type, datetime.now()),
        ),
    )

    print(f"\n\nBest Loss: {best_loss} at epoch {best_epoch}")
    torch.save(
        best_model_weights,
        os.path.join(
            "saved_models",
            "best_{}_{}_{}.pt".format(args.model_type, best_epoch, datetime.now()),
        ),
    )

    return train_losses, val_losses


def train_ssl(train_loader, val_loader, model, n_epochs, lr, resume=False):
    new_train_losses, new_val_losses = _train_val_loop(
        model, train_loader, val_loader, epochs=n_epochs, lr=lr
    )

    if resume:
        train_losses, val_losses = load_losses(saved_models_dir, args.load_losses)
    else:
        train_losses, val_losses = [], []

    train_losses.extend(new_train_losses)
    val_losses.extend(new_val_losses)

    save_losses(train_losses, val_losses, saved_models_dir, args.save_losses)

    return train_losses, val_losses, model


def main():
    if args.model_type == "base":
        model = EEmaGeBase(128, args.eeg_exclusion_channel_num, 8)
    elif args.model_type == "channelnet":
        model = EEmaGeChannelNet(
            eeg_exclusion_channel_num=args.eeg_exclusion_channel_num
        )

    if args.resume:
        resume = True
        checkpoint = args.resume

        model.load_state_dict(torch.load(checkpoint))
    else:
        resume = False

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
    val_loader = loaders["test"]

    train_losses, val_losses, model = train_ssl(
        train_loader,
        val_loader,
        model,
        n_epochs=args.epochs,
        lr=args.learning_rate,
        resume=resume,
    )

    print(f"Best Val Losses {min(val_losses):.4f}")

    writer.close()


root = os.path.dirname(__file__)
saved_models_dir = os.path.join(root, "saved_models")
if not os.path.exists(saved_models_dir):
    os.makedirs(saved_models_dir)

# Tensorboard
writer = SummaryWriter()

args = get_train_ssl_arguments()

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

torch.manual_seed(args.seed)
random.seed(args.seed)

if __name__ == "__main__":
    main()
