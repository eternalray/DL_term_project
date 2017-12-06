import os
import sys
import pickle
import timeit
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T

import STFT
from model import PresidentSing

# Usage : python main.py <inPath> <outPath> <mode>
#
#         python main.py ./audios ./converted convert
#         python main.py ./dataset ./model train

def convert(convertModel, inPath, outPath):

	if os.path.isfile(inPath):

		spectroList = STFT.transformAll(inPath)




	# read a wav file
	# wav file to Spectrogram
	spectroList = stft.transformAll(inputAudioFile, timeLength = 4)
	#print("audio length : ", len(inputSpectroList)*4)
	# input matrix (1025, 801) : frequency * time
	#print("input matrix size : ",inputSpectroList[0].shape)

	# save a spectrogram of input audio file
	
	# Forward the spectrogram
	# input spectrogram -> Model -> ouput spectrogram 

	#Error on this 
	presidentSing = PresidentSing("thisFilePathisNotUsed", 1)
	
	#Forward Path to get output spectrogram list
	outputSpectrolist = []
	for  inputSpectro in inputSpectroList:
		_,_,_,_,target_spectrogram, _, _ = presidentSing.Forward(inputSpectro)
		outputSpectrolist.append(target_spectrogram)

	#concat 'output spectrogram list' to audio array
	outputAudioArray = concatAudio(outputSpectrolist, dtype = 'spectrogram')

	#Save the output audio array to wav file
	#output file name is 'inputfilename_convert.wav'
	outputFileName = inputAudioPath[:-4] + '_' + 'convert.wav'
	librosa.output.write_wav(outputFileName, outputAudioArray,sr = 51200)

def main(inPath, outPath, mode):

	convertModel = PresidentSing(inPath, outPath, 10240)

	if mode == 'train':

		print('Train started')
		timeNow = timeit.default_timer()
		
		lossHistory = model.train()

		print('Train ended')
		print('Elapsed time : ', timeit.default_timer() - timeNow)

	elif mode == 'convert':

		convertModel.load(inPath)

		print('Convert started')
		timeNow = timeit.default_timer()

		convert(convertModel, inPath, outPath)

		print('Convert ended')
		print('Elapsed time : ', timeit.default_timer() - timeNow)

	else:

		print('Error : Mode can be "train" or "convert"')

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('inPath', help = 'Input directory of audio or dataset')
	parser.add_argument('outPath', help = 'Output directory of converted audio or model')
	parser.add_argument('mode', help = 'Mode option : <train> or <convert>')
	args = parser.parse_args()
	
	main(args.inPath, args.outPath, args.mode)
	sys.exit(0)