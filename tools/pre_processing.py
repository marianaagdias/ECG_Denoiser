import numpy as np
import pandas as pd
import os
from scipy.signal import butter, sosfilt
from scipy.stats import zscore


def ptbxl_preproc_save(data, sr=100, new_sr=None, save_dir='data500'):
	if new_sr:
		sr_ = new_sr
	else:
		sr_ = sr
	# band pass filter
	# band_pass_filter = butter(2, [1, 45], 'bandpass', fs=sr_, output='sos')

	X_aux = np.zeros_like(data)

	for i in range(np.shape(data)[0]):

		for lead in range(np.shape(data)[2]):
			if new_sr:
				sig = downsample(data[i][:, lead], sr, new_sr)
			else:
				sig = data[i][:, lead]
			# apply a band pass filter (0.05, 40hz)
			# X_aux[i][:, lead] = zscore(sosfilt(band_pass_filter, sig))
			X_aux[i][:, lead] = zscore(sig)

		np.save(save_dir + '/' + str(i) + '.npy', X_aux[i])

	return


def downsample(signal, s_rate, new_s_rate):
	# this function returns the signal downsampled to new_s_rate.
	jumps = int(s_rate/new_s_rate)
	out = signal[::jumps]
	return out


