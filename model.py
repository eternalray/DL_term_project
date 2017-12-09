import os
import sys
import time
import pickle
import random
import timeit
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data as torchData
from torch.autograd import Variable
import numpy as np

import util
import stft

class Flatten(nn.Module):

	# from cs231 assignment

	def forward(self, x):

		N, C, H, W = x.size()

		return x.view(N, -1)

class AudioLoader(torchData.Dataset):

	def __init__(self, inPath, size, target = 'trump'):

		# super(AudioLoader, self).__init__()

		dataList = list()

		files = os.listdir(inPath)
		files = [f for f in files if os.path.splitext(f)[-1] == '.pickle']
		random.shuffle(files)
		
		self.inPath = inPath
		self.fileList = files[:size]
		self.len = size
		self.target = target

	def __getitem__(self, idx):

		with open(os.path.join(self.inPath, self.fileList[idx]), 'rb') as fs:

			data, _, _ = stft.normalizeSpectro(pickle.load(fs))
			data = torch.from_numpy(data[:256,:])

		if self.target in self.fileList[idx]:

			label = 1.0

		else:

			label = 0.0

		label = torch.from_numpy(np.array([label]))

		if torch.cuda.is_available():

			data = data.cuda(1)
			label = label.cuda(1)

		return data, label

	def __len__(self):

		return self.len

class Encoder(nn.Module):

	def __init__(self):

		super(Encoder, self).__init__()
		
		# input matrix (256, 601) : frequency * time

		self.model = nn.Sequential(

			nn.Conv2d(1, 16, 3, stride = 1, padding = 1),						# (256, 601, 16)
			nn.BatchNorm2d(16),
			nn.ReLU(inplace = True),
			nn.Conv2d(16, 32, 5, stride = 3, padding = 2),						# (86, 201, 32)
			nn.BatchNorm2d(32),
			nn.ReLU(inplace = True),
			#nn.Conv2d(32, 64, 5, stride = 3, padding = (0, 1)),				# (28, 67, 64)
			#nn.BatchNorm2d(64),
			#nn.ReLU(inplace = True),
			nn.Conv2d(32, 1, 1, stride = 1, padding = 0)						# (28, 67, 1)
		)

	def forward(self, x):

		y = self.model(x)

		return y

class Decoder(nn.Module):

	def __init__(self):

		super(Decoder, self).__init__()

		self.model = nn.Sequential(

			#nn.ConvTranspose2d(1, 64, 5, stride = 3, padding = (0, 1)),			# (86, 201, 64)
			#nn.BatchNorm2d(64),
			#nn.ReLU(inplace = True),
			nn.ConvTranspose2d(1, 32, 5, stride = 3, padding = 2),				# (256, 601, 32)
			nn.BatchNorm2d(32),
			nn.LeakyReLU(negative_slope = 0.00, inplace = True),
			nn.ConvTranspose2d(32, 16, 3, stride = 1, padding = 1),				# (256, 601, 16)
			nn.BatchNorm2d(16),
			nn.LeakyReLU(negative_slope = 0.00, inplace = True),
			nn.ConvTranspose2d(16, 1, 1, stride = 1, padding = 0),				# (256, 601, 1)
		)

	def forward(self, x):

		y = self.model(x)

		return y

class Discriminator(nn.Module):

	def __init__(self):

		super(Discriminator, self).__init__()

		self.model = nn.Sequential(

			nn.Conv2d(1, 8, 3, stride = 1, padding = 1),						# (256, 601, 16)
			nn.BatchNorm2d(8),
			nn.ReLU(inplace = True),
			nn.Conv2d(8, 16, 5, stride = 3, padding = 2),						# (86, 201, 32)
			nn.BatchNorm2d(16),
			nn.ReLU(inplace = True),
			nn.Conv2d(16, 32, 5, stride = 3, padding = (0, 1)),					# (28, 67, 64)
			nn.BatchNorm2d(32),
			nn.ReLU(inplace = True),
			nn.Conv2d(32, 64, 5, stride = 3, padding = 2),						# (10, 23, 96)
			nn.BatchNorm2d(64),
			nn.ReLU(inplace = True),
			nn.Conv2d(64, 16, 5, stride = 1, padding = 1),						# (10, 23, 16)
			nn.BatchNorm2d(16),
			nn.ReLU(inplace = True),
			Flatten(),															# (10 * 23 * 16)
			nn.Linear(10 * 23 * 16, 1024, bias = True),							# (1024)
			nn.ReLU(inplace = True),
			nn.Linear(1024, 512, bias = True),									# (512)
			nn.ReLU(inplace = True),
			nn.Linear(512, 2, bias = True),										# (2)
			nn.Sigmoid()
		)			

	def forward(self, x):

		y = self.model(x)

		return y

class PresidentSing(nn.Module):

	def __init__(self, inPath, outPath, dataNum):

		super(PresidentSing, self).__init__()

		# encoder       : spectrogram (voice - any speaker) -> encoded voice code (neutral pitch, formant, tempo)
		# decoderR 		: encoded voice code (neutral pitch, formant, tempo) -> spectrogram (voice - original)
		# decoderT  	: encoded voice code (neutral pitch, formant, tempo) -> spectrogram (voice - target)
		# discriminator : spectrogram (voice) -> true or false (if target then yes, else no)
		
		# inPath		: path of dataset to train
		# outPath		: path of model to save or load
		
		self.inPath = inPath
		self.outPath = outPath
		self.dataNum = dataNum

		if torch.cuda.is_available():

			self.encoder = Encoder().cuda(1)
			self.decoderR = Decoder().cuda(1)
			self.decoderT = Decoder().cuda(1)
			self.discriminator = Discriminator().cuda(1)

		else:

			self.encoder = Encoder()
			self.decoderR = Decoder()
			self.decoderT = Decoder()
			self.discriminator = Discriminator()

	"""
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
	"""

	def convert(self, x):

		#x, mean, std = stft.normalizeSpectro(x)

		remain = x[256:,:]
		x = x[:256,:]

		x = Variable(torch.from_numpy(x), requires_grad = False)
		x = x.contiguous()
		x = x.view(1, 1, 256, 601)

		if torch.cuda.is_available():

			x = x.cuda(1)

		z = self.encoder.forward(x)
		xR = self.decoderR.forward(z)
		xT = self.decoderT.forward(z)

		if torch.cuda.is_available():

			z = z.cpu()
			xR = xR.cpu()
			xT = xT.cpu()

		z = z.data.numpy()
		xR = xR.data.numpy()
		xT = xT.data.numpy()

		#z = z.reshape(28, 67)
		z = z.reshape(86, 201)
		xR = xR.reshape(256, 601)
		xT = xT.reshape(256, 601)

		xR = xR - np.mean(remain[remain < 1.0])
		xT = xT - np.mean(remain[remain < 1.0])

		xR = np.append(xR, remain, axis = 0)
		xT = np.append(xT, remain, axis = 0)

		#xR = stft.denormalizeSpectro(xR, mean, std)
		#xT = stft.denormalizeSpectro(xT, mean, std)

		return z, xR, xT

	def train(self, learningRate = 3e-4, numEpoch = 5, numBatch = 32):

		history = list()

		self.optEncoder = optim.Adam(self.encoder.parameters(), lr = learningRate, weight_decay = 0.9)
		self.optDecoderR = optim.Adam(self.decoderR.parameters(), lr = learningRate, weight_decay = 0.9)
		self.optDecoderT = optim.Adam(self.decoderT.parameters(), lr = learningRate, weight_decay = 0.9)
		self.optDiscrim = optim.Adam(self.discriminator.parameters(), lr = learningRate, weight_decay = 0.9)

		self.lossReconstruct = nn.MSELoss()
		self.lossCycle = nn.L1Loss()
		self.lossGAN = nn.BCELoss()

		dataSet = AudioLoader(os.path.join(self.inPath, 'train'), self.dataNum)
		trainLoader = torchData.DataLoader(

			dataset = dataSet,
			batch_size = numBatch,
			shuffle = True
		)

		for epoch in range(numEpoch):

			print('Epoch ', str(epoch), ' started')
			timeNow = timeit.default_timer()

			for idx, data in enumerate(trainLoader, 0):

				lossHistory = list()
				
				# x : spectrogram
				# y : label
				x, y = data
				x = Variable(x)
				y = Variable(y.type(torch.cuda.FloatTensor), requires_grad = False)

				x = x.view(numBatch, 1, 256, 601)

				self.optEncoder.zero_grad()
				self.optDecoderR.zero_grad()
				self.optDecoderT.zero_grad()
				self.optDiscrim.zero_grad()

				# forward pass 1
				z = self.encoder.forward(x)
				xR = self.decoderR.forward(z)
				#xT = self.decoderT.forward(z)
				#zT = self.encoder.forward(xT)
				#xTR = self.decoderR.forward(zT)

				# objective1 : x == xR 			- role of autoencoder
				# objective2 : z == zT			- cycle reconstruction
				# objective3 : pX -> false		- discriminator must discriminate real voice of target and fake voice of target
				# objective4 : pT -> label 		- discriminator must discriminate target's voice and other's voice

				loss  = torch.sum(torch.abs(x - xR)) / (numBatch * 256.0 * 601.0)
				#loss = torch.sum((x - xR) ** 2) / (numBatch * 256.0 * 601.0)
				#loss = self.lossReconstruct(x, xR)
				loss.backward(retain_graph = True)
				lossHistory.append(loss.data[0])
				self.optDecoderR.step()

				if idx % 50 == 0:

					print('loss - first phase : ', loss.data[0])

				loss += torch.sum(torch.abs(z - zT)) / (numBatch * 86.0 * 201.0)
				#loss += self.lossCycle(z, zT)
				loss.backward(retain_graph = True)
				lossHistory.append(loss.data[0])
				self.optEncoder.step()

				if idx % 50 == 0:

					print('loss - second phase : ', loss.data[0])

				# forward pass 2
				pX = self.discriminator.forward(x)
				pT = self.discriminator.forward(xT)
				one = Variable(torch.Tensor([1.0]), requires_grad = False).cuda(1).expand(pX.size())

				# index 0 : Real / Fake
				# index 1 : Target / Otherwise
				# y == 1 if Target
				# y == 0 if Otherwise

				loss -= torch.sum(torch.log(pX), 0)[0] / numBatch
				loss -= torch.sum(y * torch.log(pX) + (one - y) * torch.log(one - pX), 0)[1] / numBatch
				loss -= torch.sum(torch.log(pT), 0)[0] / numBatch
				#loss -= torch.sum(torch.log(pT), 0)[1] / numBatch					# it can be a problem
				#loss += self.lossGAN(pX[0], 1)
				#loss += self.lossGAN(pX[1], y)
				#loss += self.lossGAN(pT[0], 0)
				#loss += self.lossGAN(pT[1], 1)										# it can be a problem
				loss.backward()
				lossHistory.append(loss.data[0])
				self.optDecoderT.step()
				self.optDiscrim.step()

				if idx % 50 == 0:

					print('loss - third phase : ', loss.data[0])
					print('')

				history.append((epoch, idx, lossHistory))

			print('Epoch ', str(epoch), ' finished')
			print('Elapsed time : ', str(timeit.default_timer() - timeNow))
			self.save(self.outPath, 'epoch' + str(epoch), option = 'all')
			print('')

		self.save(self.outPath, 'final', option = 'all')

		return history

	def save(self, outPath, prefix = '', option = 'all'):

		timeText = util.getTime()

		if not prefix == '':

			prefix = prefix + '_'

		if option == 'all':
			
			try:

				torch.save(self.encoder, os.path.join(outPath, timeText + prefix + 'encoder.model'))
				torch.save(self.decoderT, os.path.join(outPath, timeText + prefix + 'decoder_target.model'))
				torch.save(self.decoderR, os.path.join(outPath, timeText + prefix + 'encoder_recover.model'))
				torch.save(self.discriminator, os.path.join(outPath, timeText + prefix + 'discriminator.model'))

			except:

				print('error : save all model')

			else:

				print('successfully saved all model - ', timeText + prefix)

		elif option == 'param':

			# not implemented
			raise NOT_IMPLEMENTED

			try:

				torch.save(self.encoder.state_dict(), os.path.join(outPath, timeText + prefix + 'encoder.param'))
				torch.save(self.decoderT.state_dict(), os.path.join(outPath, timeText + prefix + 'decoder_target.param'))
				torch.save(self.decoderR.state_dict(), os.path.join(outPath, timeText + prefix + 'encoder_recover.param'))
				torch.save(self.discriminator.state_dict(), os.path.join(outPath, timeText + prefix + 'discriminator.param'))

			except:

				print('error : save parameters of model')

			else:

				print('successfully saved all parameters of model - ', timeText + prefix)

		else:

			print('error : invalid mode')

	def load(self, inPath, prefix = '', time = '', option = 'all'):

		if not prefix == '':

			prefix = prefix + '_'

		if option == 'all':
			
			try:

				# load the model files which are created lastly
				files = os.listdir(inPath)
				files.sort(reverse = True)
				timeText = files[0][:10] + '_'

				self.encoder = torch.load(os.path.join(inPath, timeText + prefix + 'encoder.model'))
				self.decoderT = torch.load(os.path.join(inPath, timeText + prefix + 'decoder_target.model'))
				self.decoderR = torch.load(os.path.join(inPath, timeText + prefix + 'encoder_recover.model'))
				self.discriminator = torch.load(os.path.join(inPath, timeText + prefix + 'discriminator.model'))

			except:

				print('error : load all model')

			else:

				print('successfully loaded all model - ', timeText + prefix)

		elif option == 'param':

			# not implemented
			raise NOT_IMPLEMENTED

			try:

				self.encoder.load_state_dict(torch.load(inPath))

			except:

				print('error : load parameters of model')

			else:

				print('successfully loaded all parameters of model - ', timeText + prefix)

		else:

			print('error : invalid mode')

		# load, parameters only
		# the_model = TheModelClass(*args, **kwargs)
		# the_model.load_state_dict(torch.load(PATH))

		# load, entire model
		# the_model = torch.load(PATH)