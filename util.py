import os
import sys
import time
import matplotlib.pyplot as plt

def getTime():

	now = time.localtime()
	timeText = str(now.tm_year)[-2:] + '%02d%02d%02d%02d_' % (now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)

	return timeText

def divideList(target, size):

	# divide given target list into sub list
	# size of sub lists is 'size'
	return [target[idx:idx + size] for idx in range(0, len(target), size)]

def plotLossHistory(lossHistory, outPath):

	fig, ax = plt.subplots()


	fig.savefig(os.path.join(os.path.dirname(outPath), getTime() + 'train_loss.png'))