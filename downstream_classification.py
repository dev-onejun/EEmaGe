import torch
from torch import nn
from torch.utils import data
from torch.utils.tensorboard.writer import SummaryWriter

from utils.train_helpers import process_large_dataset
from utils.args import get_downstream_classification_arguments
from models import EEmaGeClassifier
from datasets.perceivelab import PerceivelabClassification, ClassificationSplitter

import os, random
from copy import deepcopy
from datetime import datetime


def train(model, train_dataloader, validate_dataloader):
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4
    )
    loss_fn = nn.CrossEntropyLoss()

    best_epoch, best_accuracy, best_model_weights = 0, 0.0, None
    for epoch in range(1, args.epoch + 1):
        model.train()

        train_epoch_loss = 0.0
        train_correct = 0
        for eeg, label in process_large_dataset(train_dataloader):
            eeg, label = eeg.to(device), label.to(device)

            output = model(eeg)
            loss = loss_fn(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item() * label.size(0)
            predict = torch.argmax(output, dim=1)
            train_correct += (label == predict).sum().float()

        model.eval()
        validate_epoch_loss = 0.0
        validate_correct = 0
        with torch.no_grad():
            for eeg, label in process_large_dataset(validate_dataloader):
                eeg, label = eeg.to(device), label.to(device)

                output = model(eeg)
                loss = loss_fn(output, label)

                validate_epoch_loss += loss.item() * label.size(0)
                predict = torch.argmax(output, dim=1)
                validate_correct += (label == predict).sum().float()

        train_epoch_loss = train_epoch_loss / len(train_dataloader.dataset)
        train_accuracy = train_correct / len(train_dataloader.dataset) * 100

        validate_epoch_loss = validate_epoch_loss / len(validate_dataloader.dataset)
        validate_accuracy = validate_correct / len(validate_dataloader.dataset) * 100

        print(
            f"Epoch {epoch}\nTrain Accuracy: {train_accuracy:.4f}\tTrain Loss: {train_epoch_loss:.4f}\tValidate Accuracy: {validate_accuracy:.4f}\tValidate Loss: {validate_epoch_loss:.4f}"
        )
        writer.add_scalar("Loss/train", train_epoch_loss, epoch)
        writer.add_scalar("Loss/validate", validate_epoch_loss, epoch)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        writer.add_scalar("Accuracy/validate", validate_accuracy, epoch)

        writer.flush()

        if validate_accuracy > best_accuracy:
            best_epoch, best_accuracy = epoch, validate_accuracy
            best_model_weights = deepcopy(model.state_dict())

    torch.save(
        model.state_dict(),
        "./saved_models/{}_classification.pt".format(args.model_type),
    )

    print(f"\n\nBest Accuracy: {best_accuracy} at epoch {best_epoch}")
    torch.save(
        best_model_weights,
        os.path.join(
            "./saved_models/",
            "best_{}_{}_{}".format(args.model_type, best_epoch, datetime.now()),
        ),
    )


def main():
    model = EEmaGeClassifier(
        model_type=args.model_type,
        eeg_exclusion_channel_num=args.eeg_exclusion_channel_num,
        pretrained_model_path=args.pretrained_model_path,
    )
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    dataset = PerceivelabClassification(
        args.eeg_train_data,
        args.image_data_path,
        args.model_type,
    )
    loaders = {
        split: data.DataLoader(
            ClassificationSplitter(
                dataset,
                split_name=split,
                split_path=args.block_splits_path,
                shuffle=args.should_shuffle,
                # downstream_task=args.downstream_task,
                downstream_task=False,
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.number_workers,
        )
        for split in ["train", "val", "test"]
    }
    train_dataloader = loaders["train"]
    validate_dataloader = loaders["test"]

    train(model, train_dataloader, validate_dataloader)

    writer.close()


args = get_downstream_classification_arguments()

if args.cuda:
    assert torch.cuda.is_available() == True, "You need GPUs which support CUDA"
device = torch.device("cuda" if args.cuda else "cpu")

torch.manual_seed(args.seed)
random.seed(args.seed)

writer = SummaryWriter()

if __name__ == "__main__":
    main()
