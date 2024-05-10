import torch
from torch import nn
from torch.utils import data
from ignite.metrics import FID, InceptionScore

from datasets import Dataset, Splitter
from models import Base, EEmaGeChannelNet, BaseReconstructor, EEmaGeReconstructor
from utils.args import get_test_ssl_arguments
from utils.train_helpers import transform, process_large_dataset

import sys, os, random


# TODO: is 적용을 위한 생성 파라미터 찾아볼 것
def compute_matrix(model, test_loader):
    matrix_fid = FID()
    # matrix_is = InceptionScore()

    matrix_fid.reset()
    # matrix_is.reset()

    model.eval()
    with torch.no_grad():
        for data in process_large_dataset(test_loader):
            eeg_x, image_x, eeg_y, image_y = data
            eeg_x = eeg_x.to(device, dtype=torch.float)

            image_out = model(eeg_x)

            matrix_fid.update((image_out, image_y))
            # matrix_is.update((image_out, image_y))

    return matrix_fid.compute()  # , matrix_is.compute()


def main():
    if args.model_type == "base":
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

    dataset = Dataset(
        args.eeg_test_data, args.image_data_path, args.model_type, transform
    )
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

    # fid_ret, is_ret = compute_matrix(model, test_loader)
    fid_ret = compute_matrix(model, test_loader)
    print(f"FID: {fid_ret}")
    # print(f"IS: {is_ret}")


root = os.path.dirname(__file__)
saved_models_dir = os.path.join(root, "saved_models")
if not os.path.exists(saved_models_dir):
    sys.exit(f"Directory {saved_models_dir} does not exist.")


args = get_test_ssl_arguments()

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

torch.manual_seed(args.seed)
random.seed(args.seed)

if __name__ == "__main__":
    main()
