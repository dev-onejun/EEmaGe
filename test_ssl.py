import torch
from torch import nn
from torch.utils import data
from ignite.metrics import FID, InceptionScore

from datasets import Dataset, Splitter
from models.base import Base
from models.EEmaGeChannelNet import EEmaGeChannelNet
from models.baseReconstructor import BaseReconstructor
from models.EEmaGeReconstructor import EEmaGeReconstructor

import sys
import os
import argparse
from tqdm import tqdm

root = os.path.dirname(__file__)
saved_models_dir = os.path.join(root, "saved_models")
if not os.path.exists(saved_models_dir):
    sys.exit(f"Directory {saved_models_dir} does not exist.")


parser = argparse.ArgumentParser(description="Generate Reconstructed Image Dataset.")
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
    "--number-workers",
    "-n",
    type=int,
    default=16,
    metavar="N",
    help="number of works to load dataset (default: 16)",
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

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

torch.manual_seed(args.seed)


# TODO: is 적용을 위한 생성 파라미터 찾아볼 것
def compute_matrix(model, test_loader):
    matrix_fid = FID()
    # matrix_is = InceptionScore()

    matrix_fid.reset()
    # matrix_is.reset()

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            eeg_x, image_x, eeg_y, image_y = data
            eeg_x = eeg_x.to(device, dtype=torch.float)

            image_out = model(eeg_x)

            matrix_fid.update((image_out, image_y))
            # matrix_is.update((image_out, image_y))

    return matrix_fid.compute()  # , matrix_is.compute()


def main():
    dataset = Dataset(args.eeg_test_data, args.image_data_path, args.model_type)
    test_splitter = Splitter(
        dataset,
        split_name="test",
        split_path=args.block_splits_path,
        shuffle=False,
        downstream_task=True,
    )
    test_loader = data.DataLoader(
        test_splitter,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.number_workers,
    )

    if args.model_type == "base":
        # XXX: tran_ssl에서도 이게 맞을 듯
        # 그리고 전반적으로 train_test를 train_val로 바꾸는게 맞을 듯
        # dataset을 비롯한 대부분의 test라고 적힌것들이 사실 val이라는...
        pretrained = Base(eeg_exclusion_channel_num=args.eeg_exclusion_channel_num)
        pretrained.load_state_dict(
            torch.load(os.path.join(saved_models_dir, args.model_type + ".pt"))
        )

        model = BaseReconstructor(
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

        model = EEmaGeReconstructor()
        model.eeg_feature_extractor = pretrained.eeg_feature_extractor
        model.encoder = pretrained.encoder
        model.image_decoder = pretrained.image_decoder

    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # fid_ret, is_ret = compute_matrix(model, test_loader)
    fid_ret = compute_matrix(model, test_loader)
    print(f"FID: {fid_ret}")
    # print(f"IS: {is_ret}")


if __name__ == "__main__":
    main()
