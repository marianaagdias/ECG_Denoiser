import matplotlib.pyplot as plt
import numpy as np

import pickle
import random

import os, sys, inspect

from tools import pre_processing as pp
import scipy.signal as si
import scipy.stats as stats


# 1- Import data - noise and ECG
pcs = ['my_pc', 'server']
pc = pcs[0]

project_dir = os.getcwd()

if pc == 'my_pc':
    # webdav_dir_ecg = 'Z://DeepLearning/Data/ECG/PTB-XL/records100' # cannot access it - downloaded it
    webdav_dir_noise = 'Z://DeepLearning/Data/ECG/MIT-BIH-noise'
else:
    # webdav_dir_ecg = '/run/user/1001/gvfs/dav:host=10.136.24.49,ssl=false,prefix=%2Fowncloud%2Fremote.php%2Fwebdav/' \
    #             'DeepLearning/Data/ECG/PTB-XL/records100'
    webdav_dir_noise = '/run/user/1001/gvfs/dav:host=10.136.24.49,ssl=false,prefix=%2Fowncloud%2Fremote.php%2Fwebdav/' \
                 'DeepLearning/Data/ECG/MIT-BIH-noise'

# access the noise data
os.chdir(webdav_dir_noise)

pickle_in = open('data_noise.pickle', 'rb')
noise_in = pickle.load(pickle_in)

os.chdir(project_dir)

ma = np.array(noise_in[-1][:, 0])  # there are two "noise signals" - what is the difference?
em = np.array(noise_in[-2][:, 0])
bw = np.array(noise_in[-3][:, 0])

sample_rate = len(bw)/(30*60)  # the noise signals are 30 min long
new_s_rate = 100
num_samples = 100 * 30 * 60  # 100 Hz (ecg sampling rate)
ma = stats.zscore(si.resample(ma, num_samples))
em = stats.zscore(si.resample(em, num_samples))
bw = stats.zscore(si.resample(bw, num_samples))

noise_length = len(bw)  # 180000 (30 min)


# ecg data - there was a problem with the ecg data folder from NAS so i downloaded the data
ecg_train_directory = 'data/Y_train_all_leads/'
ecg_val_directory = 'data/Y_val_all_leads/'
ecg_test_directory = 'data/Y_test_all_leads/'

n_train = 15261
n_val = 3270
n_test = 3270

Y_train_dir = 'data/Y/Y_train/'
Y_test_dir = 'data/Y/Y_test/'
Y_val_dir = 'data/Y/Y_val/'

Y_class_train_dir = 'data/Y_class/Y_train/'
Y_class_test_dir = 'data/Y_class/Y_test/'
Y_class_val_dir = 'data/Y_class/Y_val/'

X_train_dir = 'data/X/X_train/'
X_test_dir = 'data/X/X_test/'
X_val_dir = 'data/X/X_val/'

# CREATE TRAIN SETS
for i in range(0, n_train):
    ecg = np.load(ecg_train_directory + str(i) + '.npy')
    lead_I = stats.zscore(ecg[:, 0])
    lead_II = stats.zscore(ecg[:, 1])
    lead_V2 = stats.zscore(ecg[:, 7])
    leads = [lead_I, lead_II, lead_V2]
    leads_names = ['I', 'II', 'V2']
    signal_length = len(lead_I)

    # save each lead as an Y_train sample
    with open(Y_train_dir + str(i) + '_I' + '.npy', 'wb') as f:
        np.save(f, lead_I)
    with open(Y_train_dir + str(i) + '_II' + '.npy', 'wb') as f:
        np.save(f, lead_II)
    with open(Y_train_dir + str(i) + '_V2' + '.npy', 'wb') as f:
        np.save(f, lead_V2)

    # add noise to each signal and save as an X_train sample

    # create an array with same shape as the ecg, mostly composed by 0's and with the noise signal (4 s) in a random
    # position
    noise_ma = np.zeros_like(lead_I)
    noise_em = np.zeros_like(lead_I)
    noise_bw = np.zeros_like(lead_I)

    noise_input_length = 4 * 100  # we are putting 4 seconds of noise in the signal

    st = random.randint(0, noise_length - noise_input_length)
    noise_input_ma = ma[st:st + noise_input_length]
    noise_input_em = em[st:st + noise_input_length]
    st_bw = random.randint(0, noise_length - signal_length)
    noise_input_bw = bw[st_bw:st_bw + signal_length]  # BW noise will affect the whole signal

    # add, randomly, to each lead one type of noise
    leads_n = [0, 1, 2]
    np.random.shuffle(leads_n)  # shuffle the order of the leads

    # ma noise
    start_noise = random.randint(0, signal_length - noise_input_length)
    noise_ma[start_noise:start_noise + noise_input_length] = noise_input_ma
    factor = random.uniform(0.6, 1)

    noisy_ecg_ma = leads[leads_n[0]] + factor * noise_ma
    noise_class_ma = [1, 0, 0]

    with open(X_train_dir + str(i) + '_' + str(leads_names[leads_n[0]]) + '.npy', 'wb') as f:
        np.save(f, noisy_ecg_ma)

    with open(Y_class_train_dir + str(i) + '_' + str(leads_names[leads_n[0]]) + '.npy', 'wb') as f:
        np.save(f, noise_class_ma)

    # em noise
    start_noise = random.randint(0, signal_length - noise_input_length)
    noise_em[start_noise:start_noise + noise_input_length] = noise_input_em
    factor = random.uniform(0.6, 1)

    noisy_ecg_em = leads[leads_n[1]] + factor * noise_em
    noise_class_em = [0, 1, 0]

    with open(X_train_dir + str(i) + '_' + str(leads_names[leads_n[1]]) + '.npy', 'wb') as f:
        np.save(f, noisy_ecg_em)

    with open(Y_class_train_dir + str(i) + '_' + str(leads_names[leads_n[1]]) + '.npy', 'wb') as f:
        np.save(f, noise_class_em)

    # bw noise
    factor = random.uniform(0.6, 1)

    noisy_ecg_bw = leads[leads_n[2]] + factor * noise_input_bw
    noise_class_bw = [0, 0, 1]

    with open(X_train_dir + str(i) + '_' + str(leads_names[leads_n[2]]) + '.npy', 'wb') as f:
        np.save(f, noisy_ecg_bw)

    with open(Y_class_train_dir + str(i) + '_' + str(leads_names[leads_n[2]]) + '.npy', 'wb') as f:
        np.save(f, noise_class_bw)


# CREATE TEST SETS
for i in range(0, n_test):
    ecg = np.load(ecg_test_directory + str(i) + '.npy')
    lead_I = stats.zscore(ecg[:, 0])
    lead_II = stats.zscore(ecg[:, 1])
    lead_V2 = stats.zscore(ecg[:, 7])
    leads = [lead_I, lead_II, lead_V2]
    signal_length = len(lead_I)

    lead = random.randint(0, 2)

    # save each lead as an Y_train sample
    with open(Y_test_dir + str(i) + '.npy', 'wb') as f:
        np.save(f, leads[lead])

    # add noise to each signal and save as an X_train sample

    # create an array with same shape as the ecg, mostly composed by 0's and with the noise signal (4 s) in a random
    # position
    noise = np.zeros_like(lead_I)
    noises = [ma, em, bw]
    choice = random.randint(0, 2)
    noise_choice = noises[choice]
    noise_class = [0, 0, 0]
    noise_class[choice] = 1

    noise_input_length = 4 * 100  # we are putting 4 seconds of noise in the signal

    if choice == 2: # BW noise will affect the whole signal
        st = random.randint(0, noise_length - signal_length)  # starting point from the noise signal
        noise_input = bw[st:st + signal_length]
        noise = noise_input
    # ma ou em
    else:
        st = random.randint(0, noise_length - noise_input_length)  # starting point from the noise signal
        noise_input = noise_choice[st:st + noise_input_length]
        st2 = random.randint(0, signal_length - noise_input_length)  # sample from the ecg where we will put the noise
        noise[st2:st2 + noise_input_length] = noise_input

    factor = random.uniform(0.6, 1)

    noisy_ecg = leads[lead] + factor * noise

    with open(X_test_dir + str(i) + '.npy', 'wb') as f:
        np.save(f, noisy_ecg)

    with open(Y_class_test_dir + str(i) + '.npy', 'wb') as f:
        np.save(f, noise_class)


ecg_noi = np.load(X_test_dir + str(0) + '.npy')
ecg_clean = np.load(Y_test_dir + str(0) + '.npy')
clas = np.load(Y_class_test_dir + str(0) + '.npy')

plt.plot(ecg_noi)
plt.figure()
plt.plot(ecg_clean)
print(clas)

# CREATE VALIDATION SETS
for i in range(0, n_val):
    ecg = np.load(ecg_val_directory + str(i) + '.npy')
    lead_I = stats.zscore(ecg[:, 0])
    lead_II = stats.zscore(ecg[:, 1])
    lead_V2 = stats.zscore(ecg[:, 7])
    leads = [lead_I, lead_II, lead_V2]
    signal_length = len(lead_I)

    lead = random.randint(0, 2)

    # save each lead as an Y_train sample
    with open(Y_val_dir + str(i) + '.npy', 'wb') as f:
        np.save(f, leads[lead])

    # add noise to each signal and save as an X_train sample

    # create an array with same shape as the ecg, mostly composed by 0's and with the noise signal (4 s) in a random
    # position
    noise = np.zeros_like(lead_I)
    noises = [ma, em, bw]
    choice = random.randint(0, 2)
    noise_choice = noises[choice]
    noise_class = [0, 0, 0]
    noise_class[choice] = 1

    noise_input_length = 4 * 100  # we are putting 4 seconds of noise in the signal

    if choice == 2:  # BW noise will affect the whole signal
        st = random.randint(0, noise_length - signal_length)  # starting point from the noise signal
        noise_input = bw[st:st + signal_length]
        noise = noise_input
    # ma ou em
    else:
        st = random.randint(0, noise_length - noise_input_length)  # starting point from the noise signal
        noise_input = noise_choice[st:st + noise_input_length]
        st2 = random.randint(0, signal_length - noise_input_length)  # sample from the ecg where we will put the noise
        noise[st2:st2 + noise_input_length] = noise_input

    factor = random.uniform(0.6, 1)

    noisy_ecg = leads[lead] + factor * noise

    with open(X_val_dir + str(i) + '.npy', 'wb') as f:
        np.save(f, noisy_ecg)

    with open(Y_class_val_dir + str(i) + '.npy', 'wb') as f:
        np.save(f, noise_class)


i = 0
ecg_noi = np.load(X_val_dir + str(i) + '.npy')
ecg_clean = np.load(Y_val_dir + str(i) + '.npy')
clas = np.load(Y_class_val_dir + str(i) + '.npy')

plt.plot(ecg_noi)
#plt.figure()
plt.plot(ecg_clean)
print(clas)

