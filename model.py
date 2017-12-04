import os
import sys
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T

def timeNow():

	now = time.localtime()
	timeText = str(now.tm_year)[-2:] + '%02d%02d%02d%02d_' % (now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)

	return timeTe

class Flatten(nn.Module):
    
	# from cs231 assignment

	def forward(self, x):

		N, C, H, W = x.size()

		return x.view(N, -1)

class Encoder(nn.Module):

	def __init__(self):

		# input matrix (1025, 801) : frequency * time

		self.model = nn.Sequential(

			nn.Conv2d(1, 32, 3, stride = 1, padding = 1),				# (1025, 801, 32)
			nn.BatchNorm2d(32),
			nn.ReLU(True),
			nn.Conv2d(32, 64, 5, stride = 3, padding = 1),				# (341, 267, 64)
			nn.BatchNorm2d(64),
			nn.ReLU(True),
			nn.Conv2d(64, 1, 1, stride = 1, padding = 0),				# (341, 267, 1)
			nn.Tanh()
		)

	def forward(self, x):

		y = self.model(x)

		return y

class Decoder(nn.Module):

	def __init__(self):

		self.model = nn.Sequential(

			nn.ConvTranspose2d(1, 32, 3, stride = 1, padding = 1),		# (341, 267, 32)
			nn.BatchNorm2d(32),
			nn.ReLU(True),
			nn.ConvTranspose2d(32, 64, 5, stride = 3, padding = 0),		# (1025, 801, 64)
			nn.BatchNorm2d(64),
			nn.ReLU(True),
			nn.ConvTranspose2d(64, 1, 1, stride = 1, padding = 0),		# (1025, 801, 1)
			nn.BatchNorm2d(1),
			nn.ReLU(True)
		)

	def forward(self, x):

		y = self.model(x)

		return y

class Discriminator(nn.Module):

	def __init__(self):

		self.model = nn.Sequential(

			nn.Conv2d(1, 32, 3, stride = 1, padding = 1),				# (1025, 801, 32)
			nn.BatchNorm2d(32),
			nn.ReLU(True),
			nn.Conv2d(32, 64, 5, stride = 3, padding = 1),				# (341, 267, 64)
			nn.BatchNorm2d(64),
			nn.ReLU(True),
			nn.Conv2d(64, 1, 1, stride = 1, padding = 0),				# (341, 267, 1)
			Flatten(),													# (341 * 267)
			nn.Linear(341 * 267, 4096, bias = True),					# (4096)
			nn.ReLU(True),
			nn.Linear(4096, 1024, bias = True),							# (1024)
			nn.ReLU(True),
			nn.Linear(4096, 2, bias = True),							# (2)
			nn.Softmax(True)
		)			

	def forward(self, x):

		y = self.model(x)
		_, y = torch.max(y)

		return y

def PresidentSing(nn.Module):

	def __init__(self, ):

		# Enc        : spectrogram (voice - any speaker) -> encoded voice code (neutral pitch, formant, tempo)
		# DecTarget  : encoded voice code (neutral pitch, formant, tempo) -> spectrogram (voice - target)
		# DecRecover : encoded voice code (neutral pitch, formant, tempo) -> spectrogram (voice - original)
		# Dis        : spectrogram (voice) -> true or false (if target then yes, else no)

		if torch.cuda.is_available():

			self.Enc = Encoder().cuda()
			self.DecTarget = Decoder().cuda()
			self.DecRecover = Decoder().cuda()
			self.Dis = Discriminator().cuda()

		else:

			self.Enc = Encoder()
			self.DecTarget = Decoder()
			self.DecRecover = Decoder()
			self.Dis = Discriminator()

	def forward(self, x):

		z = Enc.forward(x)
		xTarget = DecTarget.forward(z)
		zRecover = Enc.forward(xTarget)
		xRecover = DecRecover.forward(z)
		predReal = Dis.forward(x)
		predTarget = Dis.forward(xTarget)

		return z, xTarget, xRecover, xRecover, predReal, predTarget

	def train(self, learningRate = 1e-4):

		optimList = list(self.Enc.parameters()) + list(self.DecRecover.parameters())
		optimList + optimList + list(self.DecTarget.parameters()) + list(self.Dis.parameters())
		self.optimizer = optim.Adam(optimList, lr = learningRate)

		self.lossReconstruct = nn.MSELoss()
		self.lossCycle = nn.L1Loss()
		self.lossGAN = nn.CrossEntropyLoss()

		for epoch in asdfsadfads:

			# get data
			# x - spectrogram
			# y - label

			z, xTarget, xRecover, xRecover, predicted = self.forward(x)

			loss = self.lossReconstruct(x, xRecover) + self.lossCycle(z, zRecover)
			loss = loss + self.lossGAN(y, predReal) + self.lossGAN(y, 1 - predTarget)
			
			self.optimizer.zero_grad()
			loss.backward()
			optimizer.step()

	def save(self, filePath, prefix = '', option = 'param'):

		timeText = timeNow()

		if not prefix == '':

			prefix = prefix + '_'

		if option == 'all':
			
			try:

				torch.save(self.Enc, os.path.join(filePath, prefix + timeText + 'encoder.model'))
				torch.save(self.DecTarget, os.path.join(filePath, prefix + timeText + 'decoder_target.model'))
				torch.save(self.DecRecover, os.path.join(filePath, prefix + timeText + 'encoder_recover.model'))
				torch.save(self.Dis, os.path.join(filePath, prefix + timeText + 'discriminator.model'))

			except:

				print('error : save all model')

			else:

				print('successfully saved all model - ', prefix + timeText)

		elif option == 'param':

			try:

				torch.save(self.Enc.state_dict(), os.path.join(filePath, prefix + timeText + 'encoder.param'))
				torch.save(self.DecTarget.state_dict(), os.path.join(filePath, prefix + timeText + 'decoder_target.param'))
				torch.save(self.DecRecover.state_dict(), os.path.join(filePath, prefix + timeText + 'encoder_recover.param'))
				torch.save(self.Dis.state_dict(), os.path.join(filePath, prefix + timeText + 'discriminator.param'))

			except:

				print('error : save parameters of model')

			else:

				print('successfully saved all parameters of model - ', prefix + timeText)

		else:

			print('error : invalid mode')

	def load(self, filePath, prefix = '', option = 'param'):

		if not prefix == '':

			prefix = prefix + '_'

		if option == 'all':
			
			try:

				asdf
				self.Enc = torch.load(filePath)

			except:

				print('error : load all model')

			else:

				print('successfully loaded all model - ', prefix + timeText)

		elif option == 'param':

			try:

				asdfsadfsdafadsf
				self.Enc.load_state_dict(torch.load(path))

			except:

				print('error : load parameters of model')

			else:

				print('successfully loaded all parameters of model - ', prefix + timeText)

		else:

			print('error : invalid mode')

		# load, parameters only
		# the_model = TheModelClass(*args, **kwargs)
		# the_model.load_state_dict(torch.load(PATH))

		# load, entire model
		# the_model = torch.load(PATH)