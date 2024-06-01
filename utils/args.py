from argparse import ArgumentParser


def get_test_ssl_arguments():
    parser = ArgumentParser(description="Generate Reconstructed Image Dataset.")
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
        "--batch-size",
        "-b",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 32)",
    )
    parser.add_argument(
        "--epoch",
        "-e",
        type=int,
        default=100,
        help="Epoch. Default: 100",
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
        "--block-splits-path",
        type=str,
        default="./datasets/perceivelab-dataset/data/block_splits_by_image_all.pth",
        help="the path for the block splits (default: ./datasets/perceivelab-dataset/data/block_splits_by_image_all.pth)",
    )

    return parser.parse_args()


def get_train_ssl_arguments():
    parser = ArgumentParser(description="Self-supervised EEG model training.")
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
        "--eeg-val-data",
        type=str,
        default="./datasets/perceivelab-dataset/data/eeg_55_95_std.pth",
        help="the path for the root directory of the val dataset (default: ./datasets/perceivelab-dataset/data/eeg_55_95_std.pth)",
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

    return parser.parse_args()


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
