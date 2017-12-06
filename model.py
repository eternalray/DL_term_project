import os
import sys
import time
import pickle
import random
import timeit
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable

def timeNow():

	now = time.localtime()
	timeText = str(now.tm_year)[-2:] + '%02d%02d%02d%02d_' % (now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)

	return timeTe

class Flatten(nn.Module):

	# from cs231 assignment

	def forward(self, x):

		N, C, H, W = x.size()

		return x.view(N, -1)

class AudioLoader(data.Dataset):

	def __init__(self, inPath, size, target = 'trump'):

		dataList = list()

		files = os.listdir(inPath)
		files = [f for f in files if os.path.splitext(f)[-1] == '.pickle']
		random.shuffle(files)
		
		self.fileList = files[:size]
		self.len = size
		self.target = target
		#indent fixed by minuk

	def __getitem__(self, idx):

		with open(os.path.join(inPath, self.files[idx]), 'rb') as fs:

			data = torch.from_numpy(pickle.load(fs))

		if target in self.files[idx]:

			label = 1

		else:

			label = 0

		return data, label

	def __len__(self):

		return self.len

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
			nn.Sigmoid()
		)			

	def forward(self, x):

		y = self.model(x)
		_, y = torch.max(y)

		return y

class PresidentSing(nn.Module):

	def __init__(self, dataPath, dataNum):

		# encoder       : spectrogram (voice - any speaker) -> encoded voice code (neutral pitch, formant, tempo)
		# decoderR 		: encoded voice code (neutral pitch, formant, tempo) -> spectrogram (voice - original)
		# decoderT  	: encoded voice code (neutral pitch, formant, tempo) -> spectrogram (voice - target)
		# discriminator : spectrogram (voice) -> true or false (if target then yes, else no)

		self.dataPath = dataPath
		self.dataNum = dataNum

		if torch.cuda.is_available():

			self.encoder = Encoder().cuda()
			self.decoderR = Decoder().cuda()
			self.decoderT = Decoder().cuda()
			self.discriminator = Discriminator().cuda()

		else:

			self.encoder = Encoder()
			self.decoderR = Decoder()
			self.decoderT = Decoder()
			self.discriminator = Discriminator()

	def forward(self, x):

		# x 	: input
		# z 	: latent matrix of x, 								x   -> encoder  	 -> z
		# xR	: recovered spectrogram from autoencoder,			z   -> decoderR 	 -> xR
		# xT 	: spectrogram generated to target's voice,			z   -> decoderT 	 -> xT
		# zT	: latent matrix of xT,								xT  -> encoder  	 -> zT
		# xTR	: recovered target's spectrogram from autoencoder,	xTR -> encoder  	 -> zT
		# pX 	: predicted value of discriminator Real / Fake,		x   -> discriminator -> pX
		# pT 	: predicted value of discriminator True / False,	xT  -> discriminator -> pT

		z = self.encoder.forward(x)
		xR = self.decoderR.forward(z)
		xT = self.decoderT.forward(z)
		zT = self.encoder.forward(xT)
		xTR = self.decoderR.forward(zT)
		pX = self.discriminator.forward(x)
		pT = self.discriminator.forward(xT)

		return z, xR, xT, zT, xTR, pX, pT

	def convert(self, x):

		z = self.encoder.forward(x)
		xT = self.decoderT.forward(z)

		return z, xT

	def train(self, learningRate = 1e-4, numEpoch = 10, numBatch = 512):

		self.optEncoder = optim.Adam(self.encoder.parameters(), lr = learningRate)
		self.optDecoderR = optim.Adam(self.decoderR.parameters(), lr = learningRate)
		self.optDecoderT = optim.Adam(self.decoderT.parameters(), lr = learningRate)
		self.optDiscrim = optim.Adam(self.discriminator.parameters(), lr = learningRate)

		self.lossReconstruct = nn.MSELoss()
		self.lossCycle = nn.L1Loss()
		self.lossGAN = nn.BCELoss()

		dataSet = AudioLoader(self.dataPath, self.dataNum)
		trainLoader = data.DataLoader(

			dataset = dataSet,
			batch_size = numBatch,
			shuffle = True
		)

		if not os.path.exists(os.path.join(os.getcwd(), 'models')):

			os.makedirs(os.path.join(os.getcwd(), 'models'))

		for epoch in range(numEpoch):

			timeNow = timeit.default_timer()

			for idx, data in enumerate(trainLoader, 0):

				lossHistory = list()
				
				# x : spectrogram
				# y : label
				x, y = data

				self.optEncoder.zero_grad()
				self.optDecoderR.zero_grad()
				self.optDecoderT.zero_grad()
				self.optDiscrim.zero_grad()

				# forward pass 1
				z = self.encoder.forward(x)
				xR = self.decoderR.forward(z)
				xT = self.decoderT.forward(z)
				zT = self.encoder.forward(xT)
				#xTR = self.decoderR.forward(zT)

				# objective1 : x == xR 			- role of autoencoder
				# objective2 : z == zT			- cycle reconstruction
				# objective3 : pX -> false		- discriminator must discriminate real voice of target and fake voice of target
				# objective4 : pT -> label 		- discriminator must discriminate target's voice and other's voice

				loss = self.lossReconstruct(x, xR)
				loss.backward()
				lossHistory.append(loss)
				self.optDecoderR.step()

				loss += self.lossCycle(z, zT)
				loss.backward()
				lossHistory.append(loss)
				self.optEncoder.step()

				# forward pass 2
				pX = self.discriminator.forward(x)
				pT = self.discriminator.forward(xT)

				# index 0 : Real / Fake
				# index 1 : Target / Otherwise
				# y == 1 if Target
				# y == 0 if Otherwise
				loss += self.lossGAN(pX[0], 1)
				loss += self.lossGAN(pX[1], y)
				loss += self.lossGAN(pT[0], 0)
				loss += self.lossGAN(pT[1], 1)						# it can be a problem
				loss.backward()
				lossHistory.append(loss)
				self.optDecoderT.step()
				self.optDiscrim.step()

				history.append((epoch, idx, lossHistory))

			print('Epoch ', str(epoch), ' finished')
			print('Elapsed time : ', str(timeNow = timeit.default_timer() - timeNow))
			self.save(os.path.join(os.getcwd(), 'models'), 'epoch' + str(epoch), option = 'all')

		return history

	def save(self, filePath, prefix = '', option = 'param'):

		timeText = timeNow()

		if not prefix == '':

			prefix = prefix + '_'

		if option == 'all':
			
			try:

				torch.save(self.Enc, os.path.join(filePath, timeText + prefix + 'encoder.model'))
				torch.save(self.DecTarget, os.path.join(filePath, timeText + prefix + 'decoder_target.model'))
				torch.save(self.DecRecover, os.path.join(filePath, timeText + prefix + 'encoder_recover.model'))
				torch.save(self.Dis, os.path.join(filePath, timeText + prefix + 'discriminator.model'))

			except:

				print('error : save all model')

			else:

				print('successfully saved all model - ', prefix + timeText)

		elif option == 'param':

			# not implemented
			raise NOT_IMPLEMENTED

			try:

				torch.save(self.Enc.state_dict(), os.path.join(filePath, timeText + prefix + 'encoder.param'))
				torch.save(self.DecTarget.state_dict(), os.path.join(filePath, timeText + prefix + 'decoder_target.param'))
				torch.save(self.DecRecover.state_dict(), os.path.join(filePath, timeText + prefix + 'encoder_recover.param'))
				torch.save(self.Dis.state_dict(), os.path.join(filePath, timeText + prefix + 'discriminator.param'))

			except:

				print('error : save parameters of model')

			else:

				print('successfully saved all parameters of model - ', prefix + timeText)

		else:

			print('error : invalid mode')

	def load(self, filePath, prefix = '', time = '', option = 'param'):

		if not prefix == '':

			prefix = prefix + '_'

		if option == 'all':
			
			try:

				# load the model files which are created lastly
				files = os.listdir(os.path.join(os.getcwd(), 'models'))
				files.sort(reverse = True)
				textTime = files[0][:10] + '_'

				self.Enc = torch.load(os.path.join(filePath, timeText + prefix + 'encoder.model'))
				self.DecTarget = torch.load(os.path.join(filePath, timeText + prefix + 'decoder_target.model'))
				self.DecRecover = torch.load(os.path.join(filePath, timeText + prefix + 'encoder_recover.model'))
				self.Dis = torch.load(os.path.join(filePath, timeText + prefix + 'discriminator.model'))

			except:

				print('error : load all model')

			else:

				print('successfully loaded all model - ', prefix + timeText)

		elif option == 'param':

			# not implemented
			raise NOT_IMPLEMENTED

			try:

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