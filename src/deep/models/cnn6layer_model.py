import torch.nn as nn
import numpy as np
from structures.modules import PadMaxPool3d, Flatten


class CNN6Layer(nn.Module):
    """
    Classifier for a multi-class classification task

    """
    def __init__(self, input_size, dropout=0, **kwargs):
        """
        Construct a network using as entries of fc layers demographical values
        """
        super(CNN6Layer, self).__init__()

        self.flattened_shape = [-1, 128, *np.ceil(np.array(input_size)[1::] / 2**5)]

        self.features = nn.Sequential(
            nn.Conv3d(input_size[0], 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
        )

        self.classifier = nn.Sequential(
            Flatten(),

            nn.Dropout(p=dropout),
            nn.Linear(int(abs(np.prod(self.flattened_shape))), 1)
        )

    def forward(self, x, covars=None):
        """

        :param x: (FloatTensor) 5D image of size (bs, 1, 121, 145, 121)
        :return: the scores for each class
        """

        x = self.features(x)
        x = self.classifier(x)

        return x
