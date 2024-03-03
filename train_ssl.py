import torch
from torch import nn
from torch.utils import data
from torch.utils.tensorboard.writer import SummaryWriter

# from datasets. import NormalDataset
from models.autoencoders import Autoencoders, ConvAutoencoder
from train_helpers import load_losses, save_losses

import os
import argparse

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
    help="number of epochs to train (default: 10",
)
parser.add_argument(
    "--learning-rate",
    type=float,
    default=1e-3,
    metavar="L",
    help="initial learning rate (default: 1e-3",
)
parser.add_argument(
    "--load-losses",
    type=str,
    default="",
    help="path to load the latest checkpoint losses (default: None)",
)
parser.add_argument(
    "--save-losses",
    type=str,
    default="",
    help="path to save new checkpoint losses (default: None)",
)
parser.add_argument(
    "--save-name",
    type=str,
    default="checkpoint",
    help="path to save the final mode",
)

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

torch.manual_seed(args.seed)


def _eval_loss(model, test_loader, criterion):
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

            out = model(eeg_x, image_x)

            eeg_loss = criterion(out.eeg, eeg_y)
            image_loss = criterion(out.image, image_y)
            loss = eeg_loss + image_loss

            running_loss += loss.item()

        epoch_test_loss = running_loss / len(test_loader)

    return epoch_test_loss


def _train_loss(model, train_loader, optimizer, criterion):
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

        out = model(eeg_x, image_x)

        eeg_loss = criterion(out.eeg, eeg_y)
        image_loss = criterion(out.image, image_y)
        loss = eeg_loss + image_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        """ What metrics will use?
        * https://wikidocs.net/149481
            - https://pytorch.org/ignite/generated/ignite.metrics.FID.html ???

        y_pred = torch.round(torch.sigmoid(out))
        acc = (y_pred == y).sum().float() / y.shape[0]
        acc = torch.round(acc * 100)
        epoch_train_acc += acc
        """

    epoch_train_loss = total_loss / len(train_loader)
    # epoch_train_acc = epoch_train_acc / len(train_loader)

    return epoch_train_loss


def _train_test_loop(model, train_loader, test_loader, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    """
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    """
    train_losses, test_losses = [], []

    for epoch in range(1, epochs + 1):
        train_loss = _train_loss(model, train_loader, optimizer, criterion)

        train_losses.append(train_loss)

        test_loss = _eval_loss(model, test_loader, criterion)

        test_losses.append(test_loss)

        print(
            f"Epoch {epoch}, Train loss {train_loss: .4f}, Test loss {test_loss: .4f}"
        )

        scheduler.step()

        if epoch % 25 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(
                    root, "saved_models", args.save_name + "_epoch{}.pt".format(epoch)
                ),
            )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)

        writer.flush()

    torch.save(
        model.state_dict(), os.path.join(root, "saved_models", args.save_name + ".pt")
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
    model = Autoencoders(input_dim=64, hidden_dim1=32, hidden_dim2=16)

    if args.resume:
        resume = True
        checkpoint = args.resume

        model.load_state_dict(torch.load(checkpoint))
    else:
        resume = False

    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    train_dataset = False  # = NormalDataset(raw_data_files=args.train_data)
    test_dataset = False

    train_loader = data.DataLoader(
        train_dataset,
        collate_fn=train_dataset.collate_fn,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.number_workers,
    )
    test_loader = data.DataLoader(
        test_dataset,
        collate_fn=test_dataset.collate_fn,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.number_workers,
    )

    train_losses, test_losses, model = train_ssl(
        train_loader,
        test_loader,
        model,
        n_epochs=args.epochs,
        lr=args.learning_rate,
        resume=resume,
    )

    print(f"Best Test Losses {max(test_losses):.4f}")

    writer.close()


if __name__ == "__main__":
    main()
