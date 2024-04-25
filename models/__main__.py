from torchsummary import summary

if __name__ == "__main__":
    from base import Base
    #from EEmaGeChannelNet import EEmaGeChannelNet

    model = Base(128, 17, 8)
    summary(model, [(128, 440), (3, 299, 299)], device="cpu")

    """
    summary(
        EEmaGeChannelNet(eeg_exclusion_channel_num=17),
        [(1, 128, 440), (3, 299, 299)],
        device="cpu",
    )
    """
