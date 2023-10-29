import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from gru_denoiser import GRU
from utils_denoiser import configure_device
from scipy.signal import resample
import sklearn.preprocessing as pp
import tol_colors as tc
from scipy.signal import butter, sosfilt


def import_model(gpu_id=None, path_weights='best_gru_denoiser_360Hz'):
    """
        Function that prepares ECG data signals prior to using the GRU denoiser
        :param path_weights: directory to saved model's weights
        :param gpu_id: GPU device ID
    """

    # device settings
    configure_device(gpu_id)
    if gpu_id is None:
        device = 'cpu'
    else:
        device = gpu_id

    # define model's parameters
    model = GRU(n_features=1, hid_dim=64, n_layers=1, dropout=0, bidirectional=True, gpu_id=gpu_id)

    # get saved model (weights)
    model.load_state_dict(torch.load(path_weights, map_location=torch.device(device)))

    # model in the evaluation mode
    model.eval()

    return model


def prepare_ecg(signal, freq_samp=None, minmax_norm=False):
    """
    Function that prepares ECG data signals prior to using the GRU denoiser
    :param signal: ECG signal (1D)
    :param freq_samp: signal sampling frequency (only if different than 360)
    :return: signal after performing 1) resampling (if freq_samp != None) to 360 Hz, 2) Min-Max normalization and
    3) reshaping and conversion to tensor
    :param minmax_norm: (boolean) if true, perform minmax normalization instead of normalization between -1 and 1
    """
    if freq_samp:
        len_1 = len(signal)
        len_2 = int(len_1 * (360/freq_samp))
        signal = resample(signal, len_2)

    if minmax_norm:
        # Min-Max normalization
        signal_norm = pp.minmax_scale(signal)
    else:
        # normalization between -1 and 1
        signal_norm = 2*pp.minmax_scale(signal)-1

    signal_tensor = torch.from_numpy(signal_norm.reshape((1, signal_norm.shape[0], 1)))

    return signal_tensor


def clean_ecg(signal, model, figures=True, title='', postalign=True):
    """
        Function that takes an ECG signal (noisy signal) and performs the denoising, using the GRU denoiser
        :param signal: ECG signal (1D)
        :param model: imported model (denoiser)
        :param figures: (boolean) whether to display figures showing the results and comparison between input and output
        :param title: title for the figures (if figures=True)
        :param postalign: (boolean) whether to perform post alignment of the output signal
        :return: signal after performing noise removal
        :return: denoising time
    """

    # make the prediction
    start = time.time()
    sig_clean = model(signal.float())
    end = time.time()
    dur = end - start
    sig_clean_np = sig_clean.detach().numpy().reshape(signal.shape[1])

    if postalign:
        # clean very low freq noise from the output signal
        high_pass = butter(3, 0.5, 'highpass', fs=360, output='sos')
        sig_clean_np = sosfilt(high_pass, sig_clean_np)

    if figures:
        colors = list(tc.tol_cset('bright'))
        plt.figure()
        plt.plot(signal[0, :], label='input', c=colors[-1])
        plt.plot(sig_clean_np[:], label='output', c='#44AA99')
        plt.title(title)
        plt.legend()
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.subplot(211)
        plt.plot(signal[0, :])
        plt.subplot(212)
        plt.plot(sig_clean_np[:])

    return sig_clean_np, dur


def clean_ecg_segments(signal, model, seg_size=360*30, overlap=360, mean_overlap=True, figures=True, vlines=False, title='', postalign=False):
    """
        Function that takes an ECG signal (noisy signal) and performs the denoising, using the GRU denoiser
        :param signal: ECG signal (1D)
        :param model: imported model (denoiser)
        :param seg_size: (integer) length of each segment
        :param overlap: (integer) number of samples to overlap when performing the segmentation (has to be smaller than seg_size)
        :param mean_overlap: (boolean) whether to compute the mean from two consecutive segments in the overlap
        :param figures: (boolean) whether to display figures showing the results and comparison between input and output
        :param vlines: (only if figures is True) in the results plot display vlines showing the segments boundaries
        :param title: title for the results plot
        :param gpu_id: GPU device ID
        :param postalign: (boolean) whether to perform post alignment of the output signal
        :return: signal after performing noise removal
        :return: denoising time
    """

    # sig_segs = torch.split(signal.flatten(), 3600)
    # sig_segs = signal.flatten().unfold(dimension=0, size=3640, step=3600) # overlap

    sig = signal.flatten()
    signal_length = sig.size()[0]
    n_segs = round(signal_length / seg_size)  # if the last segment would be shorter than seg_size/2, then it will be
    # included in the previous segment

    # make the prediction
    start = time.time()
    sig_clean = np.empty(signal_length)

    for i in range(n_segs):
        if i == 0:
            sig_seg = sig[:seg_size + overlap]
        elif i == n_segs - 1:
            sig_seg = sig[i * seg_size - overlap:]
        else:
            sig_seg = sig[i * seg_size - overlap:(i + 1) * seg_size]
        signal_seg = sig_seg.reshape((1, sig_seg.shape[0], 1))

        sig_clean_seg = model(signal_seg.float())
        out = sig_clean_seg.detach().numpy().reshape(signal_seg.shape[1])

        if i == 0:
            sig_clean[:seg_size] = out[:- overlap]
        elif i == n_segs-1:
            if mean_overlap:
                sig_clean[i * seg_size - round(overlap / 2): i * seg_size] = \
                    (sig_clean[i * seg_size - round(overlap / 2): i * seg_size] + out[round(overlap / 2):overlap]) / 2
            sig_clean[i * seg_size:] = out[overlap:]
        else:
            sig_clean[i * seg_size - round(overlap/2) : i * seg_size] = \
                (sig_clean[i * seg_size - round(overlap/2) : i * seg_size] + out[round(overlap/2):overlap]) / 2
            sig_clean[i * seg_size:(i + 1) * seg_size] = out[overlap:]
    sig_clean_np = sig_clean.flatten()
    end = time.time()
    dur = end - start

    if postalign:
        # clean very low freq noise from the output signal
        high_pass = butter(3, 0.5, 'highpass', fs=360, output='sos')
        sig_clean_np = sosfilt(high_pass, sig_clean_np)

    if figures:
        colors = list(tc.tol_cset('bright'))
        plt.figure()
        plt.plot(signal[0, :], label='input', c=colors[-1])
        plt.plot(sig_clean_np[:], label='output', c='#44AA99')
        plt.legend()
        plt.title(title)
        if vlines:
            x_vlines = [(i+1) * seg_size for i in range(n_segs - 1)]
            plt.vlines(x_vlines, 0, 1, linewidths=2)
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.subplot(211)
        plt.plot(signal[0, :])
        plt.subplot(212)
        plt.plot(sig_clean_np[:])

    return sig_clean_np, dur





