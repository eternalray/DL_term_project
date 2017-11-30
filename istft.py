import argparse
import numpy as np
import os.path
from scipy.io import wavfile
from scipy.signal import istft

parser = argparse.ArgumentParser()
parser.add_argument("filepath", help="input directory where csv files reside")
args = parser.parse_args()

directory_path = args.filepath
loc = directory_path.rfind('/', 0, len(directory_path)-1)

prefix = directory_path[loc+1:]
file_name_idx = 1
ext = ".csv"

preblock = directory_path + "/" + prefix + "_"
file_name = preblock + str(file_name_idx) + ext
wav_data = []

print("istft started for " + prefix)
while os.path.isfile(file_name):

	with open(file_name, 'rb') as f:

		data = pickle.read(f)
	
	#data = np.loadtxt(file_name, dtype=np.complex64)
	
	print(data)
	break

	wav_data_tmp = librosa.istft(data)
	#time, wav_data_tmp = istft(data, 44100)
	wav_data = wav_data + wav_data_tmp.tolist()
	file_name_idx += 5
	file_name = preblock + str(file_name_idx) + ext

print("istft done")
wav_data_array = np.asarray(wav_data, )
wavfile.write(directory_path + "/i_" + prefix + ".wav", 44100, wav_data_array)