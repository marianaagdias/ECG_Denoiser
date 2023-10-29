import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
import pickle

import os, sys, inspect
import scipy.signal as si
import scipy.stats as stats
import sklearn.preprocessing as pp

from tools.compute_metrics import signaltonoise


def load_raw_data_local_mit(local_dir):
    data = []
    for f in [103, 105]:
        content, meta = wfdb.rdsamp(str(local_dir) + '/' + str(f))
        data.append(content)
    return data


local_dir = r'C:\Users\Catia Bastos\Downloads\mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0'
data_mit = load_raw_data_local_mit(local_dir)


#np.save('mit_bih_data/103.npy', data_mit[0])
#np.save('mit_bih_data/105.npy', data_mit[1])

#sum(data_mit[0][:,0]**2)


# access the noise data
pickle_in = open('data_noise.pickle', 'rb')
noise_in = pickle.load(pickle_in)

plt.figure()
plt.plot(np.array(noise_in[0][:, 0]))
plt.figure()
plt.plot(np.array(noise_in[1][:, 0]))
plt.figure()
plt.plot(np.array(noise_in[2][:, 0]))
plt.figure()
plt.plot(np.array(noise_in[3][:, 0]))
plt.figure()
plt.plot(np.array(noise_in[4][:, 0]))

ma = np.concatenate((np.array(noise_in[-1][:, 0]), np.array(noise_in[-1][:, 1])))  # there are two "noise signals" - what is the difference?
em = np.concatenate((np.array(noise_in[-2][:, 0]), np.array(noise_in[-2][:, 1])))
bw = np.concatenate((np.array(noise_in[-3][:, 0]), np.array(noise_in[-3][:, 1])))

sample_rate = len(bw)/(30*2*60)  # the noise signals are 30 min long
new_s_rate = 360
num_samples = 360 * 30 * 60 * 2  # 360 Hz (ecg sampling rate)
ma = pp.minmax_scale(stats.zscore(si.resample(ma, num_samples)))
em = pp.minmax_scale(stats.zscore(si.resample(em, num_samples)))
bw = pp.minmax_scale(stats.zscore(si.resample(bw, num_samples)))


def create_noisy_0db(ecg, noise):
    fac_sq = sum(ecg**2) / sum(noise**2)

    noise_to_add = noise*math.sqrt(fac_sq)

    noisy_ecg = ecg + noise_to_add

    return noisy_ecg


plt.figure()
plt.plot(data_mit[0][:, 0])
ecg = pp.minmax_scale(stats.zscore(data_mit[0][:, 0]))

bw_103 = create_noisy_0db(ecg, bw[:650000])
ma_103 = create_noisy_0db(ecg, ma[:650000])
em_103 = create_noisy_0db(ecg, em[:650000])

# check if snr is 0db
print(signaltonoise(ecg, bw_103))
print(signaltonoise(ecg, ma_103))
print(signaltonoise(ecg, em_103))

#plot
plt.figure()
plt.plot(bw_103)
plt.figure()
plt.plot(ma_103)
plt.figure()
plt.plot(em_103)

np.save('data_500/unit_test_samples/103_ma_noisy', ma_103)
np.save('data_500/unit_test_samples/103_em_noisy', em_103)
np.save('data_500/unit_test_samples/103_bw_noisy', bw_103)
np.save('data_500/unit_test_samples/103_clean', ecg)


ecg_105 = pp.minmax_scale(stats.zscore(data_mit[1][:, 0]))

bw_105 = create_noisy_0db(ecg_105, bw[:650000])
ma_105 = create_noisy_0db(ecg_105, ma[:650000])
em_105 = create_noisy_0db(ecg_105, em[:650000])

# check if snr is 0db
print(signaltonoise(ecg_105, bw_105))
print(signaltonoise(ecg_105, ma_105))
print(signaltonoise(ecg_105, em_105))

#plot
plt.figure()
plt.plot(bw_105)
plt.figure()
plt.plot(ma_105)
plt.figure()
plt.plot(em_105)

np.save('data_500/unit_test_samples/105_ma_noisy', ma_105)
np.save('data_500/unit_test_samples/105_em_noisy', em_105)
np.save('data_500/unit_test_samples/105_bw_noisy', bw_105)
np.save('data_500/unit_test_samples/105_clean', ecg_105)
