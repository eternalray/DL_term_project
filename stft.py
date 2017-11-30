import argparse
import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
from matplotlib import pyplot as plt
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("filepath", help="input path of wav file or directory")
args = parser.parse_args()

file_name = args.filepath

rate, data = wavfile.read(file_name)


left = data[:, 0]
width = 5 * rate
start = width
end = start + width
loc = file_name.rfind('/')
directory_path = file_name[:loc+1]

prefix = "spectro_" + file_name[loc+1:-4]

file_name_idx = 1
if not os.path.exists(directory_path + prefix):
	os.makedirs(directory_path + prefix)
print("stft started for " + file_name)
while end < left.shape[0]:
	spectro_file_path = directory_path + prefix + "/" + prefix + "_" + str(file_name_idx) + ".csv"
	window = left[start:end]
	freq, time, spectro = stft(window, rate)

	start += rate
	end += rate

	with open(spectro_file_path, 'wb') as f:

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