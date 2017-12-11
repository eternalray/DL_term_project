import os
import sys
import timeit
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
	#normalizedList, mean, std, maxV = stft.normalizeSpectroList(spectroList)

	normalizedList, maxV = stft.temp(spectroList)

	for normalized in normalizedList:

		#librosa.display.specshow(normalized)
		#plt.show()
		#librosa.display.specshow(normalized[:256,:])
		#plt.show()

		latent, reconst, target = model.convert(normalized)

		#librosa.display.specshow(latent)
		#plt.show()
		#librosa.display.specshow(reconst)
		#plt.show()
		#librosa.display.specshow(target)
		#plt.show()

		reconst = np.power(1.5, reconst)
		target = np.power(1.5, target)

		reconstList.append(resconst)
		targetList.append(target)

	reconst = stft.concatAudio(reconstList, dtype = 'spectrogram')
	target = stft.concatAudio(targetList, dtype = 'spectrogram')
	
	fileReconst = 'converted_reconstruct_' + os.path.basename(path)
	fileTarget = 'converted_target_' + os.path.basename(path)
	librosa.output.write_wav(os.path.join(os.path.dirname(path), fileReconst), reconst, sr = 51200)
	librosa.output.write_wav(os.path.join(os.path.dirname(path), fileTarget), target, sr = 51200)
	print('Output : ', fileReconst)
	print('Output : ', fileTarget)

def main(path, modelPath, mode):

	model = PresidentSing(path, modelPath, 4096)

	if mode == 'train':

		print('Train started')
		timeNow = timeit.default_timer()
		
		lossHistory = model.train()

		print('Train ended')
		print('Elapsed time : ', timeit.default_timer() - timeNow)

		#util.plotLossHistory(lossHistory, modelPath)

	elif mode == 'trainC':

		model.load(modelPath)

		print('Train started')
		timeNow = timeit.default_timer()
		
		lossHistory = model.train()

		print('Train ended')
		print('Elapsed time : ', timeit.default_timer() - timeNow)

		#util.plotLossHistory(lossHistory, modelPath)

	elif mode == 'convert':

		model.load(modelPath, prefix = 'final')

		print('Convert started')
		timeNow = timeit.default_timer()

		convert(model, path)

		print('Convert ended')
		print('Elapsed time : ', timeit.default_timer() - timeNow)

	else:

		print('Error : Mode can be "train" or "convert"')

if __name__ == '__main__':

	parser = argparse.ArgumentParser() 
	parser.add_argument('path', help = 'Path 1 : train - dataset directory, convert - input / output directory')
	parser.add_argument('modelPath', help = 'Path 2 : model directory')
	parser.add_argument('mode', help = 'Mode option : <train>, <convert>, or <trainC>')
	args = parser.parse_args()
	
	main(args.path, args.modelPath, args.mode)
	sys.exit(0)