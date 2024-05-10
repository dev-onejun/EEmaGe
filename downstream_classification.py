import random

from train_helpers import get_downstream_classification_arguments, process_large_dataset

import torch
from torch import nn
from torch.utils import data

from models.EEmaGe import EEmaGeClassifier
from datasets.perceivelab import PerceivelabClassification, ClassificationSplitter

from torch.utils.tensorboard.writer import SummaryWriter


def train(model, train_dataloader, validate_dataloader):
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4
    )
    loss_fn = nn.CrossEntropyLoss()

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

            train_epoch_loss += loss.item()
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

                validate_epoch_loss += loss.item()
                predict = torch.argmax(output, dim=1)
                validate_correct += (label == predict).sum().float()

        train_epoch_loss = train_epoch_loss / len(train_dataloader)
        train_accuracy = train_correct / (len(train_dataloader) * args.batch_size) * 100

        validate_epoch_loss = validate_epoch_loss / len(validate_dataloader)
        validate_accuracy = (
            validate_correct / (len(validate_dataloader) * args.batch_size) * 100
        )

        print(
            f"Epoch {epoch + 1}\nTrain Accuracy: {train_accuracy:.4f}\tTrain Loss: {train_epoch_loss:.4f}\tValidate Accuracy: {validate_accuracy:.4f}\tValidate Loss: {validate_epoch_loss:.4f}"
        )
        writer.add_scalar("Loss/train", train_epoch_loss, epoch)
        writer.add_scalar("Loss/validate", validate_epoch_loss, epoch)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        writer.add_scalar("Accuracy/validate", validate_accuracy, epoch)

        writer.flush()

    torch.save(model.state_dict(), "./saved_models/" + args.model_type + ".pt")


def main():
    model = EEmaGeClassifier(
        model_type=args.model_type,
        eeg_exclusion_channel_num=args.eeg_exclusion_channel_num,
        pretrained_model_path=args.pretrained_model_path,
    )
    model.to(device)

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
                downstream_task=args.downstream_task,
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.number_workers,
        )
        for split in ["train", "val", "test"]
    }
    train_dataloader = loaders["train"]
    validate_dataloader = loaders["val"]
    test_dataloader = loaders["test"]

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
