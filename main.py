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


from stft import transformAll
#from stft import itransform

# 여기에는 옵션에 따라 트레이닝을 해주던가, wav 파일을 주면 변형된 wav 파일을 뱉어주는 기능이 들어가야 함


def main():

	pass

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('mode', help = 'asdf')


	if mode == 'train':

		pass

	elif mode == 'use':

		pass

	else:

		pass
