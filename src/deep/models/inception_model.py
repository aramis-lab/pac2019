import torch
import torch.nn as nn

import torch.nn.functional as F


class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv3d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm3d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv3d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm3d(n3x3red),
            nn.ReLU(True),
            nn.Conv3d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm3d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv3d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm3d(n5x5red),
            nn.ReLU(True),
            nn.Conv3d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm3d(n5x5),
            nn.ReLU(True),
            nn.Conv3d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm3d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool3d(3, stride=1, padding=1),
            nn.Conv3d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm3d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 128, kernel_size=1),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
        )
        self.fc1 = nn.Linear(3456, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.adaptive_avg_pool3d(x, (3, 3, 3))
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.7, training=self.training)
        x = self.fc2(x)

        return x


class Inception3D_main(nn.Module):
    def __init__(self, training=True, **kwargs):
        super(Inception3D_main, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(True),

            nn.MaxPool3d(3, stride=2),

            nn.Conv3d(64, 64, kernel_size=1),
            nn.BatchNorm3d(64),
            nn.ReLU(True),

            nn.Conv3d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(True),

            nn.MaxPool3d(3, stride=2),
        )

        self.training = training

        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool3d(3, stride=2)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool3d((2, 3, 2), stride=1)
        self.linear = nn.Linear(1024, 1)

        self.aux1 = InceptionAux(512, 1)
        self.aux2 = InceptionAux(528, 1)

    def forward(self, x, covars = None):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)

        # For gradient injection
        if self.training:
            aux_out1 = self.aux1(out)

        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)

        # For gradient injection
        if self.training:
            aux_out2 = self.aux2(out)

        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = F.dropout(out, 0.4, training=self.training)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if self.training:
            return out, aux_out1, aux_out2
        return out


def test():
    net = Inception3D_main()
    x = torch.randn(1, 1, 121, 145, 121)
    y, _, _ = net(x)
    print(y.size())
    print(y)


def test_aux1():
    net = InceptionAux(512, 1)
    x = torch.randn(1, 512, 10, 12, 10)
    y = net(x)
    print(y.size())
    # print(y)


# test()
# test_aux1()
