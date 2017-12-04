import os
import sys
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


if torch.cuda.is_available():

	dtype = torch.cuda.FloatTensor

else:

	dtype = torch.FloatTensor

LossA = nn.CrossEntropyLoss()
LossB = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(model1.parameters()) + list(model2.parameters()), lr=1e-2)

loss = lossA(a, b) + lossB(c, d)
optimizer.zero_grad()
loss.backward()
optimizer.step()

class Encoder(nn.Module):

	def __init__(self, ):

		self.model = nn.Sequential(

			nn.Conv2d(),
			nn.ReLU(),
			nn.MaxPool2d(),



			ConvTranspose2d
		)

	def forward(self, x):

		pass

	pass

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



	pass

class Discriminator(nn.Module):

	pass

def PresidentSing(nn.Module):

	def __init__(self, ):

		# Enc        : spectrogram (voice - any speaker) -> encoded voice code (neutral pitch, formant, tempo)
		# DecTarget  : encoded voice code (neutral pitch, formant, tempo) -> spectrogram (voice - target)
		# DecRecover : encoded voice code (neutral pitch, formant, tempo) -> spectrogram (voice - original)
		# Dis        : spectrogram (voice) -> true or false (if target then yes, else no)

		self.Enc = Encoder()############.cuda()
		self.DecTarget = Decoder()
		self.DecRecover = Decoder()
		self.Dis = Discriminator()

	def forward(self, x):

		pass


	def train(self, ):


		pass

	def save(self, filePath, prefix = '', option = 'all'):

		if option == 'all':


			time time time time # attach datetime to file name
			torch.save(self.Enc, os.path.join(filePath, prefix + 'encoder.model'))

		elif option == 'param':

			pass

		else:

			print('error')
			pass

	def load(self):

		pass

		# 저장, 파라미터만
		torch.save(the_model.state_dict(), PATH)

		# 불러오기, 파라미터만
		the_model = TheModelClass(*args, **kwargs)
		the_model.load_state_dict(torch.load(PATH))

		# 저장, 모델 전체
		torch.save(the_model, PATH)

		# 불러오기, 모델 전체
		the_model = torch.load(PATH)
		pass

	pass