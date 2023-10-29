import random
import numpy as np
import os
from scipy.stats import zscore
import scipy.signal as si
from scipy.signal import butter, sosfilt

random.seed(1)

records_n = np.arange(start=0, stop=21798)
np.random.shuffle(records_n)

train = records_n[0:15261]
val = records_n[15261:18531]
test = records_n[18531:21801]

# 100 Hz
records_local_folder = 'data/records100/records100'  # downloaded the NAS folder because could not access it directly
# webdav_dir_ecg = 'Z://DeepLearning/Data/ECG/PTB-XL/'

# project_dir = os.getcwd()

j = 0
for i in val:
    print(j)
    n = np.load(records_local_folder + '/' + str(i) + '.npy')
    with open('data/Y_val_all_leads/' + str(j) + '.npy', 'wb') as f:
        np.save(f, n)
    j = j + 1

j = 0
for i in train:
    print(j)
    n = np.load(records_local_folder + '/' + str(i) + '.npy')
    with open('data/Y_train_all_leads/' + str(j) + '.npy', 'wb') as f:
        np.save(f, n)
    j = j + 1

j = 0
for i in test:
    print(j)
    n = np.load(records_local_folder + '/' + str(i) + '.npy')
    with open('data/Y_test_all_leads/' + str(j) + '.npy', 'wb') as f:
        np.save(f, n)
    j = j + 1


# 500 Hz

for file in os.listdir('data_500/Y_test_all_leads'):
    os.remove(str('data_500/Y_test_all_leads/' + file))
for file in os.listdir('data_500/Y_train_all_leads'):
    os.remove(str('data_500/Y_train_all_leads/' + file))
for file in os.listdir('data_500/Y_val_all_leads'):
    os.remove(str('data_500/Y_val_all_leads/' + file))

records_local_folder_500 = 'data500'
band_pass_filter = butter(2, [1, 45], 'bandpass', fs=500, output='sos')

j = 0
for i in val:
    n = np.load(records_local_folder_500 + '/' + str(i) + '.npy')
    sig = np.zeros((3600, 12))
    # check if the signal is suitable (criterion: the lead with the least number of peaks has to have at least 8 peaks
    # with a distance of at least 350 samples)
    peaks = []
    order = []
    for lead in range(np.shape(n)[1]):
        l = n[:, lead]
        num_peaks = len(si.find_peaks(l)[0])
        peaks.append(num_peaks)
        order.append(lead)
    leads = [ord for _, ord in sorted(zip(peaks, order))]
    # if the selection criterion is met, do the resampling and zscore of each lead and save the 12-lead signal
    if len(si.find_peaks(n[:, leads[0]], distance=300)[0]) > 8:
        for lead in range(np.shape(n)[1]):
            l = sosfilt(band_pass_filter, n[:, lead])
            sig[:, lead] = zscore(si.resample(l, 3600))
            with open('data_500/Y_val_all_leads/' + str(j) + '.npy', 'wb') as f:
                np.save(f, sig)
        j = j + 1
    else:
        print(i)

j = 0
for i in train:
    n = np.load(records_local_folder_500 + '/' + str(i) + '.npy')
    sig = np.zeros((3600, 12))
    # check if the signal is suitable (criterion: the lead with the least number of peaks has to have at least 8 peaks
    # with a distance of at least 350 samples)
    peaks = []
    order = []
    for lead in range(np.shape(n)[1]):
        l = n[:, lead]
        num_peaks = len(si.find_peaks(l)[0])
        peaks.append(num_peaks)
        order.append(lead)
    leads = [ord for _, ord in sorted(zip(peaks, order))]
    # if the selection criterion is met, do the resampling and zscore of each lead and save the 12-lead signal
    if len(si.find_peaks(n[:, leads[0]], distance=300)[0]) > 8:
        for lead in range(np.shape(n)[1]):
            l = sosfilt(band_pass_filter, n[:, lead])
            sig[:, lead] = zscore(si.resample(l, 3600))
            with open('data_500/Y_train_all_leads/' + str(j) + '.npy', 'wb') as f:
                np.save(f, sig)
        j = j + 1
    else:
        print(i)

j = 0
for i in test:
    n = np.load(records_local_folder_500 + '/' + str(i) + '.npy')
    sig = np.zeros((3600, 12))
    # check if the signal is suitable (criterion: the lead with the least number of peaks has to have at least 8 peaks
    # with a distance of at least 350 samples)
    peaks = []
    order = []
    for lead in range(np.shape(n)[1]):
        l = n[:, lead]
        num_peaks = len(si.find_peaks(l)[0])
        peaks.append(num_peaks)
        order.append(lead)
    leads = [ord for _, ord in sorted(zip(peaks, order))]
    # if the selection criterion is met, do the resampling and zscore of each lead and save the 12-lead signal
    if len(si.find_peaks(n[:, leads[0]], distance=300)[0]) > 8:
        for lead in range(np.shape(n)[1]):
            l = sosfilt(band_pass_filter, n[:, lead])
            sig[:, lead] = zscore(si.resample(l, 3600))
            with open('data_500/Y_test_all_leads/' + str(j) + '.npy', 'wb') as f:
                np.save(f, sig)
        j = j + 1
    else:
        print(i)

