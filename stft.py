import os
import sys
import pickle
import timeit
import librosa
import argparse
import numpy as np

"""
parser = argparse.ArgumentParser()
parser.add_argument('filepath', help = 'input path of wav file or directory')
args = parser.parse_args()

file_name = args.filepath
data, rate = librosa.load(file_name, mono = True)

width = 5 * rate
start = width
end = start + width
directory_path = os.path.dirname(file_name)
prefix = "spectro_" + os.path.basename(file_name)[:-4]

file_name_idx = 1

if not os.path.exists(directory_path + prefix):

	os.makedirs(directory_path + prefix)

print("stft started for " + file_name)

print(end, data.shape[0])
while end < data.shape[0]:

	spectro_file_path = os.path.join(directory_path + prefix, prefix + '_' + str(file_name_idx) + '.pickle')
	#spectro_file_path = directory_path + prefix + "/" + prefix + "_" + str(file_name_idx) + ".csv"
	window = data[start:end]
	#freq, time, spectro = stft(window, rate)
	spectro = librosa.stft(window, n_fft = 2048, hop_length = 512)

	start += rate
	end += rate

	with open(spectro_file_path, 'wb') as f:

		pickle.dump(spectro, f)

	#np.savetxt(spectro_file_path, spectro, delimiter=',')
	file_name_idx += 1

print("STFT done")
"""

def transform(filePath, timeLength = 5):

	result = list()

	data, rate = librosa.load(filePath, mono = True, sr = 160000)
	start = width = timeLength * rate
	end = start + width
	
	print("stft started for " + filePath)

	idx = 0

	while end < data.shape[0]:

		window = data[start:end]
		spectro = librosa.stft(window, n_fft = 2048, hop_length = 512)

		idx += 1
		start += rate
		end += rate
		
		result.append((idx, spectro))

	print("STFT done")

	return result

def main(inPath, outPath):

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

	

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('inPath', help = 'input path of wav file or directory')
	parser.add_argument('outPath', help = 'output path of wav file or directory')
	
	args = parser.parse_args()
	
	main(args.inPath, args.outPath)
	sys.exit(0)