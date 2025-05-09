import torch
from torch import nn
from torch.nn import MSELoss

class Generator(nn.Module):
    def __init__(self, input_channels=80, ngf=64, output_channels=80):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, ngf, kernel_size=7, padding=3),
            nn.InstanceNorm1d(ngf),
            nn.ReLU(inplace=True),

            nn.Conv1d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm1d(ngf * 2),
            nn.ReLU(inplace=True),

            nn.Conv1d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm1d(ngf * 4),
            nn.ReLU(inplace=True)
        )

        self.bottleneck = nn.Sequential(
            nn.Conv1d(ngf * 4, ngf * 4, kernel_size=3, padding=1),
            nn.InstanceNorm1d(ngf * 4),
            nn.ReLU(inplace=True),

            nn.Conv1d(ngf * 4, ngf * 4, kernel_size=3, padding=1),
            nn.InstanceNorm1d(ngf * 4),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(ngf * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(ngf),
            nn.ReLU(inplace=True),

            nn.Conv1d(ngf, output_channels, kernel_size=7, padding=3),
            nn.Sigmoid()  # assuming mel scaled to [0,1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_channels=80, ndf=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(input_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(ndf * 4, 1, kernel_size=4, padding=1)
        )

    def forward(self, x):
        return self.model(x)

# GANLoss
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True):
        super().__init__()
        self.loss = nn.MSELoss() if use_lsgan else nn.BCEWithLogitsLoss()

    def forward(self, prediction, target_is_real):
        # Make real target ——> 1
		# Make fake target ——> 0
        target = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
        return self.loss(prediction, target)

# cycle consistency loss
class CycleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, reconstruction, original):
        return self.loss(reconstruction, original)

# identity loss
class IdentityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, identity, target):
        return self.loss(identity, target)