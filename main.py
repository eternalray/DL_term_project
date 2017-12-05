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

from stft import transformAll
#from stft import itransform

#Modes - train / convert
#train : 
#convert : wav file to converted wav file

def main(mode):

	print (mode)
	pass

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('mode', help = 'Mode option have train or convert')
	args = parser.parse_args()
	
	if args.mode == 'train':
		main(args.mode)

	elif args.mode == 'convert':
		main(args.mode)

	else:
		print('mode not defined...')

	sys.exit(0)