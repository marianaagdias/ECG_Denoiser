import numpy as np
import pandas as pd
import math


def signaltonoise(sig_ori, sig_hat):
    sum_top = sum(sig_ori**2)
    sum_bot = sum((sig_hat-sig_ori)**2)
    quo = sum_top/sum_bot
    snr = 10*math.log(quo, 10)
    return snr


def signaltonoise_imp(sig, sig_noisy, sig_pred):
    return signaltonoise(sig, sig_pred) - signaltonoise(sig, sig_noisy)


def prd(sig_ori, sig_pred):
    sum_bot = sum(sig_ori ** 2)
    sum_top = sum((sig_ori - sig_pred) ** 2)
    prd = math.sqrt(sum_top/sum_bot) * 100
    return prd


def rmse(y, y_hat):
    MSE = np.square(np.subtract(y, y_hat)).mean()
    RMSE = math.sqrt(MSE)
    return RMSE


def get_sigs(i, res_file):
    classi = np.load('data/Y_class/Y_test/' + str(i) + '.npy')
    ecg_1_x = np.array(res_file['Noisy'][i * 1001 + 1:(i + 1) * (1001) - 1], dtype=float)
    ecg_1_real = np.array(res_file['Real'][i * 1001 + 1:(i + 1) * (1001) - 1], dtype=float)
    ecg_1_pred = np.array(res_file['Predicted'][i * 1001 + 1:(i + 1) * (1001) - 1], dtype=float)
    return ecg_1_real, ecg_1_x, ecg_1_pred, classi


def get_sigs_360(i, res_file):
    classi = np.load('data/Y_class/Y_test/' + str(i) + '.npy')
    ecg_1_x = np.array(res_file['Noisy'][i * 3601 + 1:(i + 1) * (3601) - 1], dtype=float)
    ecg_1_real = np.array(res_file['Real'][i * 3601 + 1:(i + 1) * (3601) - 1], dtype=float)
    ecg_1_pred = np.array(res_file['Predicted'][i * 3601 + 1:(i + 1) * (3601) - 1], dtype=float)
    return ecg_1_real, ecg_1_x, ecg_1_pred, classi


def metrics_df(results_file):
    results_test_set = []
    for i in range(0, 3267):
        ori, noisy, pred, classi = get_sigs_360(i, results_file)
        snr_in = signaltonoise(ori, noisy)
        snr_out = signaltonoise(ori, pred)
        snr_imp = signaltonoise_imp(ori, noisy, pred)
        prd_ = prd(ori, pred)
        results_test_set.append(
            {
                'SNRin': snr_in,
                'SNRout': snr_out,
                'SNRimp': snr_imp,
                'PRD': prd_
            }
        )
    results_test_set = pd.DataFrame(results_test_set)
    return results_test_set







