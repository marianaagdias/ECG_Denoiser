import matplotlib.pyplot as plt
import numpy as np
from warnings import warn

import pickle
import random

import os, sys, inspect

import scipy.signal as si
import scipy.stats as stats


def get_windows(signal, fs, window_size, overlap=None):
    """
        Function to obtain window indices with a adjustable overlapping between windows.
        :param signal: a 1-D signal
        :param fs: the sampling rate of the signal
        :param window_size: the window size in seconds
        :param hop: the hop between windows in seconds. Should be smaller than the window size. When the hop is equal
                    to window_size a sliding window with 0% overlap is performed

        :return: An index array containing all the windows and the number of samples that need to be padded the signal
                 in order to accommodate all windows.
        """
    # check validity of input
    if overlap:
        if overlap > window_size:
            raise IOError("Invalid input: Hop should be smaller or equal to the window size.")
        if overlap < 1 / fs:
            raise IOError(
                "Invalid input: The overlapping size you chose is smaller than the sampling interval T=1/fs.")
    if np.array(signal).ndim > 1:
        raise IOError("Invalid input: Only 1-D signal is allowed.")
    if window_size < 1 / fs:
        raise IOError("Invalid input: The window size you chose is smaller than the sampling interval T=1/fs.")

    # get the number of samples in the signal
    num_samples_signal = np.array(signal).size

    # calculate the number of samples that fit into a window
    num_samples_window = int(fs * window_size)

    if overlap is None:
        # calculate the number of windows that fit into a signal
        num_windows = num_samples_signal/num_samples_window
        if num_windows != int(np.round(num_windows, 0)):
            warn("The window size does not match the size of the input signal. The signal's last samples will be cut")
        windowed_signal = np.zeros((int(np.round(num_windows, 0)), num_samples_window))
        for i in range(int(np.round(num_windows, 0))):
            windowed_signal[i] = signal[i*num_samples_window:(i+1)*num_samples_window]
    else:
        print('not implemented')
        # hop = num_samples_window - overlap
        # starts = [for i in ]

    return windowed_signal


def window_and_save(diret_sig_ori, diret_sig_windowed, fs, window_size, overlap=None):
    ori = np.load(diret_sig_ori)
    output = get_windows(ori, fs, window_size, overlap)
    with open(diret_sig_windowed, 'wb') as f:
        np.save(f, output)


n_train = 15259
n_val = 3270
n_test = 3267


# DIRECTORIES

X_test = 'data_500/X/X_test'
X_train = 'data_500/X/X_train'
X_val = 'data_500/X/X_val'

Y_test = 'data_500/Y/Y_test'
Y_train = 'data_500/Y/Y_train'
Y_val = 'data_500/Y/Y_val'

X_test_ts = 'data_500/X_timesteps/X_test'
X_train_ts = 'data_500/X_timesteps/X_train'
X_val_ts = 'data_500/X_timesteps/X_val'

Y_test_ts = 'data_500/Y_timesteps/Y_test'
Y_train_ts = 'data_500/Y_timesteps/Y_train'
Y_val_ts = 'data_500/Y_timesteps/Y_val'


for i in range(0, n_train):
    for j in range(1,4):
        X_dir_ori = X_train + '/' + str(i) + '_' + str(j) + '.npy'
        Y_dir_ori = Y_train + '/' + str(i) + '_' + str(j) + '.npy'
        X_dir_wind = X_train_ts + '/' + str(i) + '_' + str(j) + '.npy'
        Y_dir_wind = Y_train_ts + '/' + str(i) + '_' + str(j) + '.npy'
        window_and_save(X_dir_ori, X_dir_wind, 1, 360)
        window_and_save(Y_dir_ori, Y_dir_wind, 1, 360)

for i in range(0, n_test):
    X_dir_ori = X_test + '/' + str(i) + '.npy'
    Y_dir_ori = Y_test + '/' + str(i) + '.npy'
    X_dir_wind = X_test_ts + '/' + str(i) + '.npy'
    Y_dir_wind = Y_test_ts + '/' + str(i) + '.npy'
    window_and_save(X_dir_ori, X_dir_wind, 1, 360)
    window_and_save(Y_dir_ori, Y_dir_wind, 1, 360)

for i in range(0, n_val):
    X_dir_ori = X_val + '/' + str(i) + '.npy'
    Y_dir_ori = Y_val + '/' + str(i) + '.npy'
    X_dir_wind = X_val_ts + '/' + str(i) + '.npy'
    Y_dir_wind = Y_val_ts + '/' + str(i) + '.npy'
    window_and_save(X_dir_ori, X_dir_wind, 1, 360)
    window_and_save(Y_dir_ori, Y_dir_wind, 1, 360)

