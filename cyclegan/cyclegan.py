from torch import nn
from torch.optim import Optimizer
from module import *
import numpy as np

device = (
	"cuda" if torch.cuda.is_available()
	else "cpu"
)

class CycleGAN(nn.Module):
	def __init__(self, generatorG, generatorF, discriminatorA, discriminatorB,
                 lambda_cycle=9.0, lambda_identity=4.0, use_lsgan=True):
		super().__init__()
		# 	generatorG A——>B
		# 	generatorF B——>A
		# discriminatorA identify A
		# discriminatorB identify B
		self.generatorG = generatorG
		self.generatorF = generatorF
		self.discriminatorA = discriminatorA
		self.discriminatorB = discriminatorB

		self.GANLoss = GANLoss(use_lsgan)
		self.CycleLoss = CycleLoss()
		self.IdentityLoss = IdentityLoss()

		self.lambda_cycle = lambda_cycle
		self.lambda_identity = lambda_identity

	def forward(self, RealA, RealB):
		# FakeA: realB ——> fakeA
		# ReconstructB fake A ——> reconstructB
		FakeA = self.generatorF(RealB)
		ReconstructB = self.generatorG(FakeA)
		# FakeB: realA ——> fakeB
		# ReconstructA fake B ——> reconstructA
		FakeB = self.generatorG(RealA)
		ReconstructA = self.generatorF(FakeB)
		return FakeA, ReconstructB, FakeB, ReconstructA

	def GeneratorLoss(self, RealA, RealB):
		FakeA, ReconstructB, FakeB, ReconstructA = self.forward(RealA, RealB)

        # GAN loss
		g_loss_G = self.GANLoss(self.discriminatorB(FakeB), True, for_discriminator=False)
		g_loss_F = self.GANLoss(self.discriminatorA(FakeA), True, for_discriminator=False)

        # cycle loss
		cycle_loss_A = self.CycleLoss(ReconstructA, RealA)
		cycle_loss_B = self.CycleLoss(ReconstructB, RealB)

        # identity loss
		identity_loss_A = self.IdentityLoss(FakeB, RealB)
		identity_loss_B = self.IdentityLoss(FakeA, RealA)

		total_loss = (
            2.0 * (g_loss_G + g_loss_F) +
            self.lambda_cycle * (cycle_loss_A + cycle_loss_B) +
            self.lambda_identity * (identity_loss_A + identity_loss_B)
        )

		return total_loss
    
	def DiscriminatorLoss(self, RealA, RealB, FakeA, FakeB, noise_std=0.01):
		noise_shape = RealA.shape
		# print(FakeA.shape)  
		noise = torch.normal(mean=0.0, std=noise_std, size=noise_shape).to(device)
		# print(noise.shape)  
		# discriminatorA: identify A is real or fake
		discriminatorA_Real_loss = self.GANLoss(self.discriminatorA(RealA+noise), True)
		discriminatorA_Fake_loss = self.GANLoss(self.discriminatorA(FakeA+noise), False)
		discriminatorA_loss = (discriminatorA_Real_loss + discriminatorA_Fake_loss) / 2
		# discriminatorB: identify B is real or fake
		discriminatorB_Real_loss = self.GANLoss(self.discriminatorB(RealB+noise), True)
		discriminatorB_Fake_loss = self.GANLoss(self.discriminatorB(FakeB+noise), False)
		discriminatorB_loss = (discriminatorB_Real_loss + discriminatorB_Fake_loss) / 2

		# discriminator_total_loss = discriminatorA_loss + discriminatorB_loss
		return discriminatorA_loss, discriminatorB_loss
