import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import neurokit2 as nk
import sklearn.preprocessing as skpp


def plot_results(i, res_file, flip=1, res_file2=None):
    classi = np.load('data/Y_class/Y_test/' + str(i) + '.npy')
    ecg_1_x = np.array(res_file['Noisy'][i*1001+1:(i+1)*(1001)-1], dtype=float)
    ecg_1_real = np.array(res_file['Real'][i*1001+1:(i+1)*(1001)-1], dtype=float)
    ecg_1_pred = np.array(res_file['Predicted'][i*1001+1:(i+1)*(1001)-1], dtype=float)
    plt.figure()
    plt.plot(ecg_1_x*(flip), '--', c='red', label='Noisy')
    plt.plot(ecg_1_real*(flip), 'b', label='Real')
    if res_file2:
        ecg_2_pred = np.array(res_file2['Predicted'][i * 1001 + 1:(i + 1) * (1001) - 1], dtype=float)
        plt.plot(ecg_2_pred*(flip), c='green', label='Result model2')
    plt.plot(ecg_1_pred*(flip), 'k', label='Result GRU')
    # plt.plot(ecg_pre_norm, 'y')
    plt.title('Sample ' + str(i), fontsize=18)
    plt.legend()
    print(classi)

