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
import argparse

import stft 
from model import PresidentSing
#from stft import itransform

#Modes
# - train : 
# - convert : 1 output

#Usage
# python main.py inputAudioFile,  mode

def main(inputAudioFile, mode):

	if mode == 'train':
		pass
	elif mode =='convert':

		# read a wav file
		# wav file to Spectrogram
		inputSpectroList = stft.transformAll(inputAudioFile, timeLength = 4)
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

	else:
		pass

	pass

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('inAudio', help = 'Mode option have train or convert')
	parser.add_argument('mode', help = 'Mode option have train or convert')
	args = parser.parse_args()
	
	if args.mode == 'convert':
		main(args.inAudio, args.mode)

	#elif args.mode == 'train':
	#	main(args.inPath, args.outPath, args.mode)
	else:
		print('Error : mode')

	sys.exit(0)