import torchsummary
from EEGChannelNet import EEGFeaturesExtractor

from Inception_v3 import ImageFeaturesExtractor
from EEmaGe import EEmaGe

torchsummary.summary(EEGFeaturesExtractor(), (1, 128, 440))
torchsummary.summary(ImageFeaturesExtractor(), (3, 299, 299))
torchsummary.summary(EEmaGe(), [(1, 128, 440), (3, 299, 299)])
