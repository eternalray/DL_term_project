import os
import sys
import timeit
import pickle
import argparse

import stft
import util
from model import PresidentSing

import librosa
import numpy as np
import matplotlib.pyplot as plt

# Usage : python main.py <inPath> <outPath> <mode>
#
#         python main.py ./audios ./model convert
#         python main.py ./dataset ./model train

def convert(model, path):

	if os.path.isfile(path):

		if os.path.splitext(path)[-1] == '.wav':

			convertFile(model, path)

	elif os.path.isdir(path):

		for ps, dirs, files in os.walk(path):

			for f in files:

				if os.path.splitext(f)[-1] == '.wav':

					convertFile(model, os.path.join(ps, f))

	else:

		print('Error : Given path is wrong')

def convertFile(model, path):

	reconstList = list()
	targetList = list()

	spectroList = stft.transformAll(path)
	normalizedList, mean, std = stft.normalizeSpectroList(spectroList)

	for normalized in normalizedList:

		#librosa.display.specshow(normalized)
		#plt.show()
		#librosa.display.specshow(normalized[:256,:])
		#plt.show()

		z, xR, xT, zT = model.convert(normalized)

		#librosa.display.specshow(z)
		#plt.show()
		#librosa.display.specshow(xR)
		#plt.show()
		#librosa.display.specshow(xT)
		#plt.show()
		#librosa.display.specshow(zT)
		#plt.show()

		#xR[:256,:] = xR[:256,:] - np.abs(np.abs(np.min(xR[:256,:])) - np.abs(np.min(xR[256:,:])))
		#xT[:256,:] = xT[:256,:] - np.abs(np.abs(np.min(xT[:256,:])) - np.abs(np.min(xT[256:,:])))

		reconst = stft.denormalizeSpectrogram(xR, mean, std)
		target = stft.denormalizeSpectrogram(xT, mean, std)

		#reconst = np.power(1.5, xR)
		#target = np.power(1.5, xT)

		#librosa.display.specshow(xR)
		#plt.show()
		#librosa.display.specshow(xT)
		#plt.show()

		reconstList.append(reconst)
		targetList.append(target)

	reconst = stft.concatAudio(reconstList, dtype = 'spectrogram')
	target = stft.concatAudio(targetList, dtype = 'spectrogram')
	
	fileReconst = 'converted_reconstruct_' + os.path.basename(path)
	fileTarget = 'converted_target_' + os.path.basename(path)
	librosa.output.write_wav(os.path.join(os.path.dirname(path), fileReconst), reconst, sr = 51200)
	librosa.output.write_wav(os.path.join(os.path.dirname(path), fileTarget), target, sr = 51200)
	print('Output : ', fileReconst)
	print('Output : ', fileTarget)

def testDiscriminator(model, path, target):

	result = list()
	ground = list()
	count = list()


	if os.path.isdir(path):

		for ps, dirs, files in os.walk(path):

			for f in files:

				if os.path.splitext(f)[-1] == '.pickle':

					with open(os.path.join(ps, f), 'rb') as fs:

						spectro = pickle.load(fs)
						pred = model.testDiscriminator(spectro)

						if pred[1] > 0.5:

							count.append(1.0)

							if target in os.path.splitext(f)[0]:

								result.append(1.0)
								ground.append(1.0)

							else:

								result.append(0.0)

						else:

							if not target in os.path.splitext(f)[0]:

								result.append(1.0)

							else:

								result.append(0.0)
								ground.append(1.0)

	else:

		print('Error : Given path is wrong')

	acc = np.sum(result) / len(result)
	truth = np.sum(ground)

	return acc, np.sum(count), truth, len(files)

def main(path, modelPath, mode):

	#model = PresidentSing(path, modelPath, 4096)
	model = PresidentSing(path, modelPath, 12288)

	if mode == 'train':

		print('Train started')
		timeNow = timeit.default_timer()
		
		lossHistory = model.train()

		print('Train ended')
		print('Elapsed time : ', timeit.default_timer() - timeNow)

		#util.plotLossHistory(lossHistory, modelPath)
		util.saveLossHistory(lossHistory, os.getcwd())

	elif mode == 'trainC':

		model.load(modelPath)

		print('Train started')
		timeNow = timeit.default_timer()
		
		lossHistory = model.train()

		print('Train ended')
		print('Elapsed time : ', timeit.default_timer() - timeNow)

		#util.plotLossHistory(lossHistory, modelPath)
		util.saveLossHistory(lossHistory, modelPath)

	elif mode == 'convert':

		model.load(modelPath, prefix = 'final')

		print('Convert started')
		timeNow = timeit.default_timer()

		convert(model, path)

		print('Convert ended')
		print('Elapsed time : ', timeit.default_timer() - timeNow)

	elif mode == 'discrim':

		model.load(modelPath, prefix = 'final')

		print('Test started')
		timeNow = timeit.default_timer()

		result = testDiscriminator(model, path, 'trump')

		print('Test ended')
		print(result)
		print('Elapsed time : ', timeit.default_timer() - timeNow)

	else:

		print('Error : Mode can be "train" or "convert" or "discrim"')

if __name__ == '__main__':

	parser = argparse.ArgumentParser() 
	parser.add_argument('path', help = 'Path 1 : train - dataset directory, convert - input / output directory')
	parser.add_argument('modelPath', help = 'Path 2 : model directory')
	parser.add_argument('mode', help = 'Mode option : <train>, <convert>, or <trainC>')
	args = parser.parse_args()
	
	main(args.path, args.modelPath, args.mode)
	sys.exit(0)
