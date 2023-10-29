import numpy as np
import scipy.stats as stats
from statistics import variance
from scipy.signal import welch
import neurokit2 as nk
import matplotlib.pyplot as plt
import tol_colors as tc


def variation(y):
    return stats.variation(y)


def var(y):
    return variance(y)


def hos(sig):
    return np.abs(stats.skew(sig)) * (stats.kurtosis(sig)/5)


def snr_no_GT(sig):
    return np.std(sig)**2/(np.std(np.abs(sig))**2)


def kur(sig):
    return stats.kurtosis(sig)


def skew(sig):
    return stats.skew(sig)


def relative_power(sig):
    f, Pxx = welch(sig, fs=360, detrend='linear')
    f_15 = f < 16
    f_5 = f > 4
    freqs_5_15_Hz = f_15 & f_5

    relative_power = sum(Pxx[freqs_5_15_Hz]) / sum(Pxx)

    return relative_power


def rr_timeseries(sig, fig=True):
    _, results_bef = nk.ecg_peaks(sig.flatten(), sampling_rate=360)
    rr_dist = np.diff(results_bef["ECG_R_Peaks"])
    if fig:
        plt.figure()
        plt.plot(rr_dist)
    return rr_dist


def outliers_limits(sig):
    _, results_bef = nk.ecg_peaks(sig.flatten(), sampling_rate=360)

    rr_dist = np.diff(results_bef["ECG_R_Peaks"])

    quartiles = np.quantile(rr_dist, [0, 0.25, 0.5, 0.75])
    q1 = quartiles[1]
    q3 = quartiles[3]

    ipr = q3 - q1

    up_lim = q1 + 1.25*ipr
    low_lim = q3 - 1.25*ipr

    return up_lim, low_lim


def rr_outliers(sig, up_lim, low_lim, check_outl=True, plots=True, title=''):
    _, results_bef = nk.ecg_peaks(sig.flatten(), sampling_rate=360)

    peaks_ind = results_bef["ECG_R_Peaks"]

    rr_dist = np.diff(peaks_ind)
    rr_dist_diff = np.diff(rr_dist)
    rr_seg = rr_dist / 360
    rr_bpm = 60 / rr_seg

    out_up = np.where(rr_dist > up_lim)[0]
    out_low = np.where(rr_dist < low_lim)[0]

    if check_outl:
        remove = []
        i = 0
        for o in out_up:
            if abs(rr_dist_diff[o - 1]) < 50:
                remove.append(i)
            i = i + 1
        out_up = np.delete(out_up, remove)

        remove = []
        i = 0
        for o in out_low:
            if abs(rr_dist_diff[o - 1]) < 50:
                remove.append(i)
            i = i + 1
        out_low = np.delete(out_low, remove)

    colors = list(tc.tol_cset('bright'))
    if plots:
        plt.figure()
        plt.plot(sig.flatten(), c=colors[-1])
        plt.plot(peaks_ind, sig.flatten()[peaks_ind], 'o', c=colors[-2], label='peaks')
        plt.plot(peaks_ind[out_up], sig.flatten()[peaks_ind][out_up], '*', c=colors[0])
        plt.plot(peaks_ind[out_low], sig.flatten()[peaks_ind][out_low], '*', c=colors[1])
        plt.title(title)
        plt.legend()

        plt.figure()
        plt.plot(rr_bpm, c=colors[-1])
        plt.plot(out_up, rr_bpm[out_up], 'o', c=colors[0])
        plt.plot(out_low, rr_bpm[out_low], 'o', c=colors[1])
        plt.title(title)

    return peaks_ind, rr_dist, [out_up, out_low]


def missing_peaks(rr_dist, out_up):
    """
    :param rr_dist: timeseries with the distance (in samples) between peaks
    :param out_up: vector with indexes from "rr_dist" that contain outliers (too high distance)
    :return: number of peaks that were not detected
    """
    miss = 0
    median = np.quantile(rr_dist, 0.5)
    for outl in out_up:
        if rr_dist[outl] > median * 2:
            add = round((rr_dist[outl] - median) / median)
        else:
            add = 1
        miss = miss + add
    return miss


def wrong_detection(out_low):
    """
    :param out_low: vector with indexes from "rr_dist" that contain outliers (too short distance)
    :return: number of peaks that were wrongly detected
    """
    wrong = len(out_low)
    return wrong


def samples_to_bpm(rr_dist):
    rr_seg = rr_dist / 360
    rr_bpm = 60 / rr_seg
    return rr_bpm
