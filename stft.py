import os
import pickle
import argparse
import librosa
import librosa.display
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
from matplotlib import pyplot as plt

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

		print('asdf')
		pickle.dump(spectro, f)

	#np.savetxt(spectro_file_path, spectro, delimiter=',')
	file_name_idx += 1

print("stft done")

	

'''
plt.pcolormesh(time, freq, spectro)

plt.ylabel("Frequency [Hz]")
plt.xlabel("time [sec]")
plt.ylim(0,5000)
#plt.xlim(10,20)
plt.show()
'''