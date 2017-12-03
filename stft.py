import os
import sys
import pickle
import timeit
import librosa
import argparse
import numpy as np

def divideList(target, size):

	# divide given target list into sub list
	# size of sub lists is 'size'
	return [target[idx:idx + size] for idx in range(0, len(target), size)]

def transformExtract(filePath, timeLength = 5, size = 100):

	result = list()

	# load audio
	audio, rate = librosa.load(filePath, mono = True, sr = 160000)
	
	# remove each 10% of foremost, hindmost from audio
	# size of window is (sampling rate) * (time in second)
	start = int(audio.shape[0] * 0.1)
	end = int(audio.shape[0] * 0.9)
	windowSize = timeLength * rate

	# assertion
	assert end > start + windowSize and audio.shape[0] > end + windowSize, 'Audio is too short!'

	# select random point of audio
	selected = numpy.random.randint(start, end, size = size)

	# STFT for selected and divided audio
	for idx in selected:

		window = audio[idx : idx + windowSize]
		spectro = librosa.stft(window, n_fft = 2048, hop_length = 512)

		result.append(spectro)

	# return STFT to train, val, test set
	return result[: int(size * 0.6)], result[int(size * 0.6) : int(size * 0.8)], result[int(size * 0.8):]

def transformAll(filePath, timeLength = 5):

	result = list()

	# load audio
	audio, rate = librosa.load(filePath, mono = True, sr = 160000)
	
	# remove each 10% of foremost, hindmost from audio
	# size of window is (sampling rate) * (time in second)
	start = int(audio.shape[0] * 0.1)
	end = int(audio.shape[0] * 0.9)
	windowSize = timeLength * rate

	# use 80% of audio
	selected = audio[start : end]

	# STFT for divided audio
	for window in divideList(selected, windowSize):

		spectro = librosa.stft(window, n_fft = 2048, hop_length = 512)

		result.append(spectro)
		
	return result

"""
def itransform(filePath):

	with open(file_name, 'rb') as f:

		data = pickle.read(f)

	audio = librosa.istft(data)

	return audio

def saveAudio(filePath):
"""

def main(inPath, outPath, mode = 'continuous'):

	if mode == 'continuous':

		if not os.path.exists(outPath):

			os.makedirs(outPath)

		for path, dirs, files in os.walk(inPath):

			for f in files:

				if os.path.splitext(f)[-1] == '.wav':

					try:

						spectroList = transformAll(os.path.join(path, f))

						for spectro in spectroList:

							outFile = 'spectro_' + str(spectro[0]) + '_' + os.path.splitext(f)[0] + '.pickle'

							with open(os.path.join(outPath, outFile), 'wb') as fs:

								pickle.dump(spectro[1], fs)

					except:

						continue

	elif mode == 'extraction':

		if not os.path.exists(outPath):

			os.makedirs(outPath)

		if not os.path.exists(os.path.join(outPath, 'train')):

			os.makedirs(os.path.join(outPath, 'train'))

		if not os.path.exists(os.path.join(outPath, 'val')):

			os.makedirs(os.path.join(outPath, 'val'))

		if not os.path.exists(os.path.join(outPath, 'test')):

			os.makedirs(os.path.join(outPath, 'test'))

		for path, dirs, files in os.walk(inPath):

			for f in files:

				if os.path.splitext(f)[-1] == '.wav':

					try:

						train, val, test = transformExtract(os.path.join(path, f))

						for spectro in train:

							outFile = 'spectro_' + str(spectro[0]) + '_' + os.path.splitext(f)[0] + '.pickle'

							with open(os.path.join(os.path.join(outPath, 'train'), outFile), 'wb') as fs:

								pickle.dump(spectro[1], fs)

						for spectro in val:

							outFile = 'spectro_' + str(spectro[0]) + '_' + os.path.splitext(f)[0] + '.pickle'

<<<<<<< HEAD
							with open(os.path.join(os.path.join(outPath, 'val'), outFile), 'wb') as fs:

								pickle.dump(spectro[1], fs)

						for spectro in test:

							outFile = 'spectro_' + str(spectro[0]) + '_' + os.path.splitext(f)[0] + '.pickle'

							with open(os.path.join(os.path.join(outPath, 'test'), outFile), 'wb') as fs:

								pickle.dump(spectro[1], fs)

					except:

						continue

	else:

		print('Error : mode')
=======
	if os.path.isdir(inPath):

		for path, dirs, files in os.walk(inPath):

			for f in files:

				if os.path.splitext(f)[-1] == '.wav':

					spectroList = transform(os.path.join(path, f))

					for spectro in spectroList:

						if not os.path.exists(outPath):

							os.makedirs(outPath)

						outFile = 'spectro_' + str(spectro[0]) + '_' + os.path.splitext(f)[0] + '.pickle'

						with open(os.path.join(outPath, outFile), 'wb') as fs:

							pickle.dump(spectro[1], fs)	

	elif os.path.isfile(inPath):

		if os.path.splitext(inPath)[-1] == '.wav':

			spectroList = transform(inPath)

			for spectro in spectroList:

				if not os.path.exists(outPath):

					os.makedirs(outPath)

				outFile = 'spectro_' + str(spectro[0]) + '_' + os.path.splitext(inPath)[0] + '.pickle'

				with open(os.path.join(outPath, outFile), 'wb') as fs:

					pickle.dump(spectro[1], fs)

	else:

		raise

	
>>>>>>> ffa5dd55c883ef408131922ee5135e95c21a4152

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('inPath', help = 'input path of wav file or directory')
<<<<<<< HEAD
	parser.add_argument('outPath', help = 'input path of wav file or directory')
	parser.add_argument('mode', help = 'continous or extraction')
=======
	parser.add_argument('outPath', help = 'output path of wav file or directory')
>>>>>>> ffa5dd55c883ef408131922ee5135e95c21a4152
	
	args = parser.parse_args()
	
	main(args.inPath, args.outPath, args.mode)
	sys.exit(0)