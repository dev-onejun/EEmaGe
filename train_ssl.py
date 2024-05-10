import torch
from torch import nn
from torch.utils import data
from torch.utils.tensorboard.writer import SummaryWriter

from datasets import Dataset, Splitter
from models.base import Base
from models.EEmaGeChannelNet import EEmaGeChannelNet
from train_helpers import load_losses, save_losses

import os, argparse, random
from tqdm import tqdm

root = os.path.dirname(__file__)
saved_models_dir = os.path.join(root, "saved_models")
if not os.path.exists(saved_models_dir):
    os.makedirs(saved_models_dir)

# Tensorboard
writer = SummaryWriter()

parser = argparse.ArgumentParser(description="Self-supervised EEG model training.")
parser.add_argument(
    "--no-cuda",
    action="store_true",
    default=False,
    help="disable CUDA training",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    metavar="S",
    help="random seed (default: 42)",
)
parser.add_argument(
    "--resume",
    type=str,
    default="",
    help="path to latest checkpoint (default: None)",
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
    "--number-workers",
    "-n",
    type=int,
    default=16,
    metavar="N",
    help="number of works to load dataset (default: 16)",
)
parser.add_argument(
    "--epochs",
    "-e",
    type=int,
    default=10,
    metavar="N",
    help="number of epochs to train (default: 10)",
)
parser.add_argument(
    "--learning-rate",
    type=float,
    default=3e-4,
    metavar="L",
    help="initial learning rate (default: 3e-4)",
)
parser.add_argument(
    "--load-losses",
    type=str,
    default="",
    help="path to load the checkpoint losses (default: None)",
)
parser.add_argument(
    "--save-losses",
    type=str,
    default="",
    help="path to save a checkpoint losses (default: None)",
)
parser.add_argument(
    "--eeg-train-data",
    type=str,
    default="./datasets/perceivelab-dataset/data/eeg_55_95_std.pth",
    help="the path for the root directory of the training dataset (default: ./datasets/perceivelab-dataset/data/eeg_55_95_std.pth)",
)
parser.add_argument(
    "--eeg-test-data",
    type=str,
    default="./datasets/perceivelab-dataset/data/eeg_55_95_std.pth",
    help="the path for the root directory of the test dataset (default: ./datasets/perceivelab-dataset/data/eeg_55_95_std.pth)",
)
parser.add_argument(
    "--image-data-path",
    type=str,
    default="./datasets/imagenet-dataset/train",
    help="the path for the root directory of the image dataset (default: ./datasets/imagenet-dataset/train)",
)
parser.add_argument(
    "--model-type",
    type=str,
    default="base",
    help="model selection option (base / channelnet are supported so far) (default: base)",
)
parser.add_argument(
    "--eeg-exclusion-channel-num",
    type=int,
    default=0,
    help="The number of unrelated EEG Channels (default: 0) (recommend: 0|17)",
)
parser.add_argument(
    "--step-size",
    type=int,
    default=10,
    help="A step size in the StepLR scheduler",
)
parser.add_argument(
    "--block-splits-path",
    type=str,
    default="./datasets/perceivelab-dataset/data/block_splits_by_image_all.pth",
    help="the path for the block splits (default: ./datasets/perceivelab-dataset/data/block_splits_by_image_all.pth)",
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

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

torch.manual_seed(args.seed)
random.seed(args.seed)


def _eval_loss(model, test_loader, eeg_criterion, image_criterion):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
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
            loss = eeg_loss + image_loss

            running_loss += loss.item()

        epoch_test_loss = running_loss / len(test_loader)

    return epoch_test_loss


def _train_loss(model, train_loader, optimizer, eeg_criterion, image_criterion):
    model.train()

    total_loss = 0.0
    # epoch_train_acc = 0.0

    for i, data in enumerate(tqdm(train_loader)):
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
        loss = eeg_loss + image_loss

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


def _train_test_loop(model, train_loader, test_loader, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # eeg_criterion = mean_absolute_average_error
    eeg_criterion = nn.MSELoss()
    image_criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=0.9
    )

    train_losses, test_losses = [], []

    for epoch in range(1, epochs + 1):
        train_loss = _train_loss(
            model, train_loader, optimizer, eeg_criterion, image_criterion
        )

        train_losses.append(train_loss)

        test_loss = _eval_loss(model, test_loader, eeg_criterion, image_criterion)

        test_losses.append(test_loss)

        print(
            f"Epoch {epoch}, \t Train loss {train_loss: .4f}, \t Test loss {test_loss: .4f}"
        )

        scheduler.step()

        if epoch % args.step_size == 0:
            torch.save(
                model.state_dict(),
                os.path.join(
                    root, "saved_models", args.model_type + "_epoch{}.pt".format(epoch)
                ),
            )
            save_losses(
                train_losses,
                test_losses,
                saved_models_dir,
                args.save_losses + "_epoch{}".format(epoch),
            )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)

        writer.flush()

    torch.save(
        model.state_dict(), os.path.join(root, "saved_models", args.model_type + ".pt")
    )

    return train_losses, test_losses


def train_ssl(train_loader, test_loader, model, n_epochs, lr, resume=False):
    new_train_losses, new_test_losses = _train_test_loop(
        model, train_loader, test_loader, epochs=n_epochs, lr=lr
    )

    if resume:
        train_losses, test_losses = load_losses(saved_models_dir, args.load_losses)
    else:
        train_losses, test_losses = [], []

    train_losses.extend(new_train_losses)
    test_losses.extend(new_test_losses)

    save_losses(train_losses, test_losses, saved_models_dir, args.save_losses)

    return train_losses, test_losses, model


def main():
    if args.model_type == "base":
        model = Base(128, args.eeg_exclusion_channel_num, 8)
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

    dataset = Dataset(args.eeg_train_data, args.image_data_path, args.model_type)
    loaders = {split: data.DataLoader(Splitter(dataset, split_name=split, split_path=args.block_splits_path, shuffle=args.should_shuffle, downstream_task=args.downstream_task), batch_size=args.batch_size, shuffle=True, num_workers=args.number_workers) for split in ["train", "val", "test"]}

    train_loader = loaders["train"]
    test_loader = loaders["test"]
    # val_loader = loaders["val"]

    train_losses, test_losses, model = train_ssl(
        train_loader,
        test_loader,
        model,
        n_epochs=args.epochs,
        lr=args.learning_rate,
        resume=resume,
    )

    print(f"Best Test Losses {min(test_losses):.4f}")

    writer.close()


if __name__ == "__main__":
    main()
