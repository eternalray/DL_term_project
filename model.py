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

class Encoder(nn.Module):

	def __init__(self, ):

		# input matrix (1025, 801) : frequency * time

		self.model = nn.Sequential(

			nn.Conv2d(),
			nn.ReLU(),
			nn.MaxPool2d(),



			ConvTranspose2d
		)

	def forward(self, x):

		y = self.model(x)

		return y

class Decoder(nn.Module):

	def __init__(self, ):

		self.model = nn.Sequential(

			nn.ConvTranspose2d(),
			nn.ReLU(),
			nn.Tanh(),
		)

	def forward(self, x):

		y = self.model(x)

		return y

class Discriminator(nn.Module):

	def __init__(self, ):

		self.model = nn.Sequential(

			nn.ConvTranspose2d(),
			nn.ReLU(),
			nn.Tanh(),
		)

	def forward(self, x):

		y = self.model(x)

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
		xRecover = DecRecover.forward(z)
		xTarget = DecTarget.forward(z)
		predicted = Dis.forward(xTarget)

		return z, xRecover, xTarget, predicted

	def train(self, ):

		optimList = list(self.Enc.parameters()) + list(self.DecRecover.parameters())
		optimList + optimList + list(self.DecTarget.parameters()) + list(self.Dis.parameters())
		self.optimizer = optim.Adam(optimList, lr = 1e-4)

		self.lossA = nn.CrossEntropyLoss()
		self.lossB = nn.CrossEntropyLoss()



		for epoch in asdfsadfads:

			loss = self.lossA(a, b) + self.lossB(c, d)
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