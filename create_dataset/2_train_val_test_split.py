import random
import numpy as np
import os
from scipy.stats import zscore
import scipy.signal as si
from scipy.signal import butter, sosfilt

# in this script, the records from PTB-XL (saved as numpy files from running "1_get_save_data_from_db.py", are split in
# training, validation and testing sets and are pre-processed to be used in the development of the model.

# the pre-processing step consists of: (i) checking if the signal is suitable, (ii) down-sampling the data to 360 Hz,
# (iii) applying a second-order bandpass butterworth filter, with a high-pass cutoff frequency of 1 Hz and a low-pass
# cutoff frequency of 45Hz to eliminate high-frequency noise

random.seed(1)

# randomly select 15261, 3270, 3270 records for the training, validation and testing sets, respectively
records_n = np.arange(start=0, stop=21798)
np.random.shuffle(records_n)

train = records_n[0:15261]
val = records_n[15261:18531]
test = records_n[18531:21801]

# directories where each set's records are going to be stored
Y_test_dir= 'data/Y_test_all_leads'
Y_trai_dir= 'data/Y_train_all_leads'
Y_val_dir = 'data/Y_val_all_leads'

for file in os.listdir(Y_test_dir):
    os.remove(str(Y_test_dir + '/' + file))
for file in os.listdir(Y_trai_dir):
    os.remove(str(Y_trai_dir + '/' + file))
for file in os.listdir(Y_val_dir):
    os.remove(str(Y_val_dir + '/' + file))

records_local_folder_500 = 'data/ptb_xl_500hz'
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
            with open(Y_val_dir + '/' + str(j) + '.npy', 'wb') as f:
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
            with open(Y_trai_dir + '/' + str(j) + '.npy', 'wb') as f:
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
            with open(Y_test_dir + '/' + str(j) + '.npy', 'wb') as f:
                np.save(f, sig)
        j = j + 1
    else:
        print(i)

