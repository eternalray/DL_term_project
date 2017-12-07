import os
import sys
import timeit
import argparse

import stft
import util
from model import PresidentSing

import librosa
import numpy as np

# Usage : python main.py <inPath> <outPath> <mode>
#
#         python main.py ./audios ./model convert
#         python main.py ./dataset ./model train

def convert(convertModel, path, mode = 'target'):

	if os.path.isfile(path):

		if os.path.splitext(path)[-1] == '.wav':

			fileName, audio = convertFile(convertModel, path, mode)

	elif os.path.isdir(path):

		for ps, dirs, files in os.walk(path):

			for f in files:

				if os.path.splitext(f)[-1] == '.wav':

					fileName, audio = convertFile(convertModel, os.path.join(ps, f), mode)

	else:

		print('Error : Given path is wrong')

def convertFile(convertModel, path, mode = 'target', show = False):

	audioList = list()

	spectroList = stft.transformAll(path)
	normalizedList = stft.normalizeSpectroList(spectroList)

	for normalized, mean, std in normalizedList:

		if mode == 'target':

			_, _, converted = convertModel.convert(normalized)
			fileName = 'converted_target_' + os.path.basename(path)

		elif mode == 'reconstruct':

			_, converted, _ = convertModel.convert(normalized)
			fileName = 'converted_reconstruct_' + os.path.basename(path)

		else:

			print('Mode can be "target" or "reconstruct"')

		converted = stft.denormalizeSpectro(converted, mean, std)
		convertedAudio = stft.griffinLim(converted)
		audioList.append(convertedAudio)

	audio = stft.concatAudio(audioList)
	dirName = os.path.dirname(path)
	
	librosa.output.write_wav(os.path.join(dirName, fileName), audio, sr = 51200)
	print('Output : ', fileName)

	if show:

		original, _ = librosa.load(path, mono = True, sr = 51200)
		
		stft.showSpectrogram(original)
		stft.showSpectrogram(audio)

	return fileName, audio

def main(path, modelPath, mode):

	convertModel = PresidentSing(path, modelPath, 1024)

	if mode == 'train':

		print('Train started')
		timeNow = timeit.default_timer()
		
		lossHistory = convertModel.train()

		print('Train ended')
		print('Elapsed time : ', timeit.default_timer() - timeNow)

		#util.plotLossHistory(lossHistory, modelPath)

	elif mode == 'trainC':

		convertModel.load(modelPath)

		print('Train started')
		timeNow = timeit.default_timer()
		
		lossHistory = convertModel.train()

		print('Train ended')
		print('Elapsed time : ', timeit.default_timer() - timeNow)

		util.plotLossHistory(lossHistory, modelPath)

	elif mode == 'convert':

		convertModel.load(modelPath, prefix = 'final')

		print('Convert started')
		timeNow = timeit.default_timer()

		convert(convertModel, path)
		convert(convertModel, path, 'reconstruct')

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