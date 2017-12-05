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

def transformExtract(filePath, timeLength = 4, size = 100):

	result = list()

	# load audio
	audio, rate = librosa.load(filePath, mono = True, sr = 51200)
	
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
		spectro = librosa.stft(window, n_fft = 2048, hop_length = 256, win_length = 1024)

		# need padding last window, make shape (1025, 801)
		if spectro.shape[1] < 801:

			spectro = np.lib.pad(spectro, ((0, 0), (0, 801 - spectro.shape[1])), 'constant', constant_values = 0.0)

		result.append(np.abs(spectro))

	# return STFT to train, val, test set
	return result[: int(size * 0.6)], result[int(size * 0.6) : int(size * 0.8)], result[int(size * 0.8):]

def transformAll(filePath, timeLength = 4):

	result = list()

	# load audio
	audio, rate = librosa.load(filePath, mono = True, sr = 51200)
	
	# remove each 10% of foremost, hindmost from audio
	# size of window is (sampling rate) * (time in second)
	start = int(audio.shape[0] * 0.1)
	end = int(audio.shape[0] * 0.9)
	windowSize = timeLength * rate

	# use 80% of audio
	selected = audio[start : end]

	# use 100% of audio
	#selected = audio

	# STFT for divided audio
	for window in divideList(selected, windowSize):

		spectro = librosa.stft(window, n_fft = 2048, hop_length = 256, win_length = 1024)

		# need padding last window, make shape (1025, 801)
		if spectro.shape[1] < 801:

			spectro = np.lib.pad(spectro, ((0, 0), (0, 801 - spectro.shape[1])), 'constant', constant_values = 0.0)

		result.append(np.abs(spectro))
		
	return result

def griffinLim(spectro, iterN = 50):

	# reference : https://github.com/andabi/deep-voice-conversion/blob/master/tools/audio_utils.py

	phase = np.pi * np.random.rand(*spectro.shape)
	spec = spectro * np.exp(1.0j * phase)

	for i in range(iterN):

		audio = librosa.istft(spec, hop_length = 256, win_length = 1024, length = 204800)

		if i < iterN - 1:

			spec = librosa.stft(audio, n_fft = 2048, hop_length = 256, win_length = 1024)
			_, phase = librosa.magphase(spec)
			spec = spectro * np.exp(1.0j * np.angle(phase))

	return audio

def concatAudio(dataList, dtype = 'audio'):

	# dataList must be sorted
	audio = list()

	if dtype == 'audio':

		for data in dataList:

			audio = audio + list(data)

	elif dtype == 'spectrogram':

		for data in dataList:

			audio = audio + list(griffinLim(data))

	return np.array(audio)

def main(inPath, outPath, mode = 'continuous'):

	if mode == 'continuous':

		if not os.path.exists(outPath):

			os.makedirs(outPath)

		for path, dirs, files in os.walk(inPath):

			for f in files:

				if os.path.splitext(f)[-1] == '.wav':

					try:

						spectroList = transformAll(os.path.join(path, f))

						for idx, spectro in enumerate(spectroList):

							outFile = 'spectro_' + str(idx) + '_' + os.path.splitext(f)[0] + '.pickle'

							with open(os.path.join(outPath, outFile), 'wb') as fs:

								pickle.dump(spectro, fs)

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

						for idx, spectro in enumerate(train):

							outFile = 'spectro_' + str(idx) + '_' + os.path.splitext(f)[0] + '.pickle'

							with open(os.path.join(os.path.join(outPath, 'train'), outFile), 'wb') as fs:

								pickle.dump(spectro, fs)

						for idx, spectro in enumerate(val):

							outFile = 'spectro_' + str(idx) + '_' + os.path.splitext(f)[0] + '.pickle'

							with open(os.path.join(os.path.join(outPath, 'val'), outFile), 'wb') as fs:

								pickle.dump(spectro, fs)

						for idx, spectro in enumerate(test):

							outFile = 'spectro_' + str(idx) + '_' + os.path.splitext(f)[0] + '.pickle'

							with open(os.path.join(os.path.join(outPath, 'test'), outFile), 'wb') as fs:

								pickle.dump(spectro, fs)

					except:

						continue

	else:

		print('Error : mode')

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('inPath', help = 'input path of wav file or directory')
	parser.add_argument('outPath', help = 'input path of wav file or directory')
	parser.add_argument('mode', help = 'continous or extraction')
	
	args = parser.parse_args()
	
	main(args.inPath, args.outPath, args.mode)
	sys.exit(0)