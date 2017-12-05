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

	def __getitem__(self, idx):

		with open(os.path.join(inPath, self.files[idx]), 'rb') as fs:

			data = torch.from_numpy(pickle.load(fs))

		if target in self.files[idx]:

			label = True

		else:

			label = False

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
			nn.Softmax(True)
		)			

	def forward(self, x):

		y = self.model(x)
		_, y = torch.max(y)

		return y

def PresidentSing(nn.Module):

	def __init__(self, dataPath, dataNum):

		# Enc        : spectrogram (voice - any speaker) -> encoded voice code (neutral pitch, formant, tempo)
		# DecTarget  : encoded voice code (neutral pitch, formant, tempo) -> spectrogram (voice - target)
		# DecRecover : encoded voice code (neutral pitch, formant, tempo) -> spectrogram (voice - original)
		# Dis        : spectrogram (voice) -> true or false (if target then yes, else no)

		self.dataPath = dataPath
		self.dataNum = dataNum

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

		z = 
		xTarget = DecTarget.forward(z)
		zRecover = Enc.forward(xTarget)
		xRecover = DecRecover.forward(z)
		predReal = Dis.forward(x)
		predTarget = Dis.forward(xTarget)

		return z, xTarget, xRecover, xRecover, predReal, predTarget

	def train(self, learningRate = 1e-4, numEpoch = 5, numBatch = 128):

		# optimList = list(self.Enc.parameters()) + list(self.DecRecover.parameters())
		# optimList + optimList + list(self.DecTarget.parameters()) + list(self.Dis.parameters())
		# self.optimizer = optim.Adam(optimList, lr = learningRate)

		optEnc
		optDecTarget
		optDecRecover

		self.lossReconstruct = nn.MSELoss()
		self.lossCycle = nn.L1Loss()
		self.lossGAN = nn.CrossEntropyLoss()

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

				# x : spectrogram
				# y : label
				x, y = data

				z, xTarget, xRecover, xRecover, predicted = self.forward(x)

				loss = self.lossReconstruct(x, xRecover) + self.lossCycle(z, zRecover)
				loss = loss + self.lossGAN(y, predReal) + self.lossGAN(y, 1 - predTarget)
				
				self.optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			print('Epoch ', str(epoch), ' finished, Elapsed time : ', str(timeNow = timeit.default_timer() - timeNow))
			self.save(os.path.join(os.getcwd(), 'models'), 'epoch' + str(epoch), option = 'all')

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