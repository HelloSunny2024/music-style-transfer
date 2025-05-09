import torch
from torch import nn
from torch.nn import MSELoss


# residual block
class ResidualBlock(nn.Module):
	def __init__(self, input_channels, output_channels):
		super().__init__()
		self.block = nn.Sequential(
			nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1),
			nn.InstanceNorm1d(output_channels),
			nn.ReLU(True),
			nn.Conv1d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1),
			nn.InstanceNorm1d(output_channels)
		)
		self.relu = nn.ReLU(True)

	def forward(self, x):
		y = self.block(x)
		return self.relu(x + y)


# convolutional block
class ConvolutionalBlock(nn.Module):
	def __init__(self, input_channels, output_channels):
		super().__init__()
		self.block = nn.Sequential(
			nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=2, padding=1),
			nn.InstanceNorm1d(output_channels),
			nn.ReLU(True)
		)

	def forward(self, x):
		x = self.block(x)
		return x


# deconvolutional block
class DeconvolutionalBlock(nn.Module):
	def __init__(self, input_channels, output_channels):
		super().__init__()
		self.block = nn.Sequential(
			nn.ConvTranspose1d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=2,
							   padding=1, output_padding=1),
			nn.InstanceNorm1d(output_channels),
			nn.ReLU(True)
		)

	def forward(self, x):
		x = self.block(x)
		return x


class UpsampleConvBlock(nn.Module):
	def __init__(self, input_channels, output_channels):
		super().__init__()
		self.block = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='nearest'),  # Nearest neighbor upsampling
			nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1),
			nn.InstanceNorm1d(output_channels),
			nn.ReLU(True)
		)

	def forward(self, x):
		x = self.block(x)
		return x


# generator for convert A ——> B OR B ——> A
class Generator(nn.Module):
	def __init__(self, ngf, input_channels, output_channels):
		super().__init__()
		self.inputlayer = nn.Sequential(
			nn.Conv1d(in_channels=input_channels, out_channels=ngf, kernel_size=7, padding=3),
			nn.InstanceNorm1d(ngf),
			nn.ReLU(True)
		)
		# self.conv1 = ConvolutionalBlock(ngf, ngf * 2)

		self.conv1 = ConvolutionalBlock(ngf, ngf)
		self.conv2 = ConvolutionalBlock(ngf*2, ngf * 4)
		self.conv3 = ConvolutionalBlock(ngf * 4, ngf * 4)

		# residual blocks
		# self.res1 = ResidualBlock(ngf * 2, ngf * 2)
		self.res1 = ResidualBlock(ngf * 4, ngf * 4)
		self.res2 = ResidualBlock(ngf * 4, ngf * 4)
		self.res3 = ResidualBlock(ngf * 4, ngf * 4)
		self.res4 = ResidualBlock(ngf * 4, ngf * 4)
		self.res5 = ResidualBlock(ngf * 4, ngf * 4)
		self.res6 = ResidualBlock(ngf * 4, ngf * 4)
		self.res7 = ResidualBlock(ngf * 4, ngf * 4)
		self.res8 = ResidualBlock(ngf * 4, ngf * 4)
		self.res9 = ResidualBlock(ngf * 4, ngf * 4)

		# deconvolutional layer
		# self.deconv1 = DeconvolutionalBlock(ngf * 2, ngf)
		# self.deconv1 = UpsampleConvBlock(ngf * 2, ngf)
		self.deconv1 = UpsampleConvBlock(ngf * 4, ngf * 2)
		self.deconv2 = UpsampleConvBlock(ngf * 2, ngf)
		self.deconv3 = UpsampleConvBlock(ngf, ngf)

		# 	output layer
		self.outputlayer = nn.Sequential(
			nn.Conv1d(in_channels=ngf, out_channels=output_channels, kernel_size=7, padding=3),
			nn.Tanh()
		)

	def forward(self, x):
		x = self.inputlayer(x)
		# print(x.shape)
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.res1(x)
		x = self.res2(x)
		x = self.res3(x)
		# x = self.res4(x)
		# x = self.res5(x)
		# x = self.res6(x)
		# x = self.res7(x)
		# x = self.res8(x)
		# x = self.res9(x)
		x = self.deconv1(x)
		x = self.deconv2(x)
		x = self.deconv3(x)
		x = self.outputlayer(x)
		x = x[:, :, :2580]
		return x


# discriminator for classification A OR B
class Discriminator(nn.Module):
	def __init__(self, input_channels, ndf):
		# super(DiscriminatorWithAttention, self).__init__()
		super().__init__()
		self.model = nn.Sequential(
			# First convolutional layer
			nn.Conv1d(in_channels=input_channels, out_channels=ndf, kernel_size=4, stride=2, padding=1),
			nn.LeakyReLU(0.2, inplace=True),
			# Second convolutional layer
			nn.Conv1d(in_channels=ndf, out_channels=ndf * 2, kernel_size=4, stride=2, padding=1),
			nn.InstanceNorm1d(ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# Third convolutional layer
			# nn.Conv1d(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=4, stride=2, padding=1),
			# nn.InstanceNorm1d(ndf * 4),
			# nn.LeakyReLU(0.2, inplace=True),
			# output layer
            # nn.Conv1d(in_channels=ndf * 4, out_channels=1, kernel_size=4, padding=3),
			nn.Conv1d(in_channels=ndf * 2, out_channels=1, kernel_size=4, padding=3),
			nn.Sigmoid()
		)

	def forward(self, x):
		output = self.model(x)
		return output


# GANLoss
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True):
        super().__init__()
        self.loss = nn.MSELoss() if use_lsgan else nn.BCELoss()

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