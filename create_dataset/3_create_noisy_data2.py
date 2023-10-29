import matplotlib.pyplot as plt
import numpy as np

import pickle
import random

import os, sys, inspect

# from tools import pre_processing as pp
import scipy.signal as si
import scipy.stats as stats

import sklearn.preprocessing as pp


# access the noise data
pickle_in = open('data_noise.pickle', 'rb')
noise_in = pickle.load(pickle_in)


ma = np.concatenate((np.array(noise_in[-1][:, 0]), np.array(noise_in[-1][:, 1])))  # there are two "noise signals" - what is the difference?
em = np.concatenate((np.array(noise_in[-2][:, 0]), np.array(noise_in[-2][:, 1])))
bw = np.concatenate((np.array(noise_in[-3][:, 0]), np.array(noise_in[-3][:, 1])))

sample_rate = len(bw)/(30*2*60)  # the noise signals are 30 min long
new_s_rate = 100
num_samples = 100 * 30 * 60 * 2  # 100 Hz (ecg sampling rate)
ma = stats.zscore(si.resample(ma, num_samples))
em = stats.zscore(si.resample(em, num_samples))
bw = stats.zscore(si.resample(bw, num_samples))

noise_length = len(bw)  # 180000 (30 min)

ma_train = ma[:28000]
ma_val = ma[28000:32000]
ma_test = ma[32000:]

em_train = em[:28000]
em_val = em[28000:32000]
em_test = em[32000:]

bw_train = bw[:28000]
bw_val = bw[28000:32000]
bw_test = bw[32000:]


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


# Empty Datasets directories (to change the datasets)
for file in os.listdir(X_val_dir):
    os.remove(str(X_val_dir + '/' + file))
for file in os.listdir(Y_val_dir):
    os.remove(str(Y_val_dir + '/' + file))

for file in os.listdir(X_train_dir):
    os.remove(str(X_train_dir + '/' + file))
for file in os.listdir(Y_train_dir):
    os.remove(str(Y_train_dir + '/' + file))

for file in os.listdir(X_test_dir):
    os.remove(str(X_test_dir + '/' + file))
for file in os.listdir(Y_test_dir):
    os.remove(str(Y_test_dir + '/' + file))


def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return 20*np.log10(abs(np.where(sd == 0, 0, m/sd)))


# CREATE TRAIN SETS
new_i = 0
for i in range(0, n_train):
    ecg = np.load(ecg_train_directory + str(i) + '.npy')

    # band_pass_filter = si.butter(2, [1, 45], 'bandpass', fs=100, output='sos')

    peaks = []
    ord = []
    for j in range(0, 12):
        num_peaks = len(si.find_peaks(ecg[:, j])[0])
        peaks.append(num_peaks)
        ord.append(j)
    leads = [ord for _,ord in sorted(zip(peaks, ord))]

    lead_I = stats.zscore(ecg[:, leads[0]])
    lead_II = stats.zscore(ecg[:, leads[1]])
    lead_V2 = stats.zscore(ecg[:, leads[2]])

    if len(si.find_peaks(lead_I, distance=70)[0]) > 8:

        # apply a band pass filter (0.05, 40hz)
        # lead_I = si.sosfilt(band_pass_filter, lead_I)
        # lead_II = si.sosfilt(band_pass_filter, lead_II)
        # lead_V2 = si.sosfilt(band_pass_filter, lead_V2)

        leads = [lead_I, lead_II, lead_V2]
        leads_names = ['I', 'II', 'V2']
        signal_length = len(lead_I)

        # save each lead as an Y_train sample
        with open(Y_train_dir + str(new_i) + '_I' + '.npy', 'wb') as f:
            np.save(f, lead_I)
        with open(Y_train_dir + str(new_i) + '_II' + '.npy', 'wb') as f:
            np.save(f, lead_II)
        with open(Y_train_dir + str(new_i) + '_V2' + '.npy', 'wb') as f:
            np.save(f, lead_V2)

        # add noise to each signal and save as an X_train sample

        # create an array with same shape as the ecg, mostly composed by 0's and with the noise signal (4 s) in a random
        # position
        noise_ma = np.zeros_like(lead_I)
        noise_em = np.zeros_like(lead_I)
        noise_bw = np.zeros_like(lead_I)

        noise_input_length = random.randint(2, 5) * 100  # we are putting 2 to 5 seconds of noise in the signal

        noise_train_length = len(ma_train)
        st = random.randint(0, noise_train_length - noise_input_length)
        noise_input_ma = ma_train[st:st + noise_input_length]
        noise_input_em = em_train[st:st + noise_input_length]
        st_bw = random.randint(0, noise_train_length - signal_length)
        noise_input_bw = bw_train[st_bw:st_bw + signal_length]  # BW noise will affect the whole signal

        # add, randomly, to each lead one type of noise
        leads_n = [0, 1, 2]
        np.random.shuffle(leads_n)  # shuffle the order of the leads

        # ma noise
        start_noise = random.randint(0, signal_length - noise_input_length)
        noise_ma[start_noise:start_noise + noise_input_length] = noise_input_ma
        factor = random.uniform(0.7, 1.3)

        noisy_ecg_ma = leads[leads_n[0]] + factor * noise_ma
        noise_class_ma = [1, 0, 0]

        with open(X_train_dir + str(new_i) + '_' + str(leads_names[leads_n[0]]) + '.npy', 'wb') as f:
            np.save(f, noisy_ecg_ma)

        with open(Y_class_train_dir + str(new_i) + '_' + str(leads_names[leads_n[0]]) + '.npy', 'wb') as f:
            np.save(f, noise_class_ma)

        # em noise
        start_noise = random.randint(0, signal_length - noise_input_length)
        noise_em[start_noise:start_noise + noise_input_length] = noise_input_em
        factor = random.uniform(1, 1.5)

        noisy_ecg_em = leads[leads_n[1]] + factor * noise_em
        noise_class_em = [0, 1, 0]

        with open(X_train_dir + str(new_i) + '_' + str(leads_names[leads_n[1]]) + '.npy', 'wb') as f:
            np.save(f, noisy_ecg_em)

        with open(Y_class_train_dir + str(new_i) + '_' + str(leads_names[leads_n[1]]) + '.npy', 'wb') as f:
            np.save(f, noise_class_em)

        # bw noise
        # for this sample, we will add bw noise OR bw and one of the other types of noise OR bw and both of the others

        select_noise = ['bw', 'bw + ma', 'bw + em', 'ma + em', 'bw + ma + em']
        np.random.shuffle(select_noise)
        selection = select_noise[0]
        # print(selection)
        factor_bw = random.uniform(0.7, 1.3)
        factor_ma = random.uniform(0.7, 1.3)
        factor_em = random.uniform(1, 1.5)

        if selection == 'bw':
            noisy_ecg_bw = leads[leads_n[2]] + factor_bw * noise_input_bw
            noise_class_bw = [0, 0, 1]
        elif selection == 'bw + ma':
            noisy_ecg_bw = leads[leads_n[2]] + factor_bw * noise_input_bw \
                           + factor_ma * noise_ma
            noise_class_bw = [1, 0, 1]
        elif selection == 'bw + em':
            noisy_ecg_bw = leads[leads_n[2]] + factor_bw * noise_input_bw \
                           + factor_em * noise_em
            noise_class_bw = [0, 1, 1]
        elif selection == 'ma + em':
            noisy_ecg_bw = leads[leads_n[2]] \
                           + factor_ma * noise_ma + factor_em * noise_em
            noise_class_bw = [1, 1, 0]
        elif selection == 'bw + ma + em':
            noisy_ecg_bw = leads[leads_n[2]] + factor_bw * noise_input_bw \
                           + factor_ma * noise_ma + factor_em * noise_em
            noise_class_bw = [1, 1, 1]

        with open(X_train_dir + str(new_i) + '_' + str(leads_names[leads_n[2]]) + '.npy', 'wb') as f:
            np.save(f, noisy_ecg_bw)

        with open(Y_class_train_dir + str(new_i) + '_' + str(leads_names[leads_n[2]]) + '.npy', 'wb') as f:
            np.save(f, noise_class_bw)

        new_i = new_i + 1
    else:
        print(ecg_train_directory + str(i) + '.npy')


# CREATE TEST SETS
for i in range(0, n_test):
    ecg = np.load(ecg_test_directory + str(i) + '.npy')
    leads_n = random.sample([i for i in range(0, 12)], 3)
    lead_I = stats.zscore(ecg[:, leads_n[0]])  # (ecg[:, 0])
    lead_II = stats.zscore(ecg[:, leads_n[1]])  # (ecg[:, 1])
    lead_V2 = stats.zscore(ecg[:, leads_n[2]])  # (ecg[:, 7])
    leads = [lead_I, lead_II, lead_V2]
    signal_length = len(lead_I)

    lead = random.randint(0, 2)

    # save the lead as an Y_train sample
    with open(Y_test_dir + str(i) + '.npy', 'wb') as f:
        np.save(f, leads[lead])

    # add noise to each signal and save as an X_train sample

    # create an array with same shape as the ecg, mostly composed by 0's and with the noise signal (4 s) in a random
    # position
    noise_input_length = random.randint(2, 5) * 100  # 2 to 5 seconds
    noise_ma = np.zeros_like(lead_I)
    noise_em = np.zeros_like(lead_I)
    noise_bw = np.zeros_like(lead_I)

    noise_test_length = len(ma_test)
    st = random.randint(0, noise_test_length - noise_input_length)  # starting point from the noise signal for the em/ma
    st_bw = random.randint(0, noise_test_length - signal_length)
    st2_ma = random.randint(0, signal_length - noise_input_length)  # sample from the ecg where we will put the noise
    st2_em = random.randint(0, signal_length - noise_input_length)

    noise_input_ma = ma_test[st:st + noise_input_length]
    noise_input_em = em_test[st:st + noise_input_length]
    noise_input_bw = bw_test[st_bw:st_bw + signal_length]  # BW noise will affect the whole signal

    noise_ma[st2_ma:st2_ma + noise_input_length] = noise_input_ma
    noise_em[st2_em:st2_em + noise_input_length] = noise_input_em

    factor_bw = random.uniform(0.7, 1.3)
    factor_ma = random.uniform(0.7, 1.3)
    factor_em = random.uniform(1, 1.5)

    choice = random.randint(0, 2)  # 0 - ma; 1 - em; 2 - bw or any combination

    if choice == 0:  # ma
        noise_class = [1, 0, 0]
        noisy_ecg = leads[lead] + factor_ma * noise_ma
    elif choice == 1:  # em
        noise_class = [0, 1, 0]
        noisy_ecg = leads[lead] + factor_em * noise_em
    else:
        select_noise = ['bw', 'bw + ma', 'bw + em', 'ma + em', 'bw + ma + em']
        np.random.shuffle(select_noise)
        selection = select_noise[0]
        if selection == 'bw':
            noise_class = [0, 0, 1]
            noisy_ecg = leads[lead] + factor_bw * noise_input_bw
        elif selection == 'bw + ma':
            noise_class = [1, 0, 1]
            noisy_ecg = leads[lead] + factor_bw * noise_input_bw + factor_ma * noise_ma
        elif selection == 'bw + em':
            noise_class = [0, 1, 1]
            noisy_ecg = leads[lead] + factor_bw * noise_input_bw + factor_em * noise_em
        elif selection == 'ma + em':
            noise_class = [1, 1, 0]
            noisy_ecg = leads[lead] + factor_ma * noise_ma + factor_em * noise_em
        elif selection == 'bw + ma + em':
            noise_class = [1, 1, 1]
            noisy_ecg = leads[lead] + factor_bw * noise_input_bw + factor_ma * noise_ma + factor_em * noise_em

    with open(X_test_dir + str(i) + '.npy', 'wb') as f:
        np.save(f, noisy_ecg)

    with open(Y_class_test_dir + str(i) + '.npy', 'wb') as f:
        np.save(f, noise_class)


# CREATE VALIDATION SETS
for i in range(0, n_val):
    ecg = np.load(ecg_val_directory + str(i) + '.npy')
    leads_n = random.sample([i for i in range(0, 12)], 3)
    lead_I = stats.zscore(ecg[:, leads_n[0]])  # (ecg[:, 0])
    lead_II = stats.zscore(ecg[:, leads_n[1]])  # (ecg[:, 1])
    lead_V2 = stats.zscore(ecg[:, leads_n[2]])  # (ecg[:, 7])
    leads = [lead_I, lead_II, lead_V2]
    signal_length = len(lead_I)

    lead = random.randint(0, 2)

    # save the lead as an Y_train sample
    with open(Y_val_dir + str(i) + '.npy', 'wb') as f:
        np.save(f, leads[lead])

    # add noise to each signal and save as an X_train sample

    # create an array with same shape as the ecg, mostly composed by 0's and with the noise signal (4 s) in a random
    # position
    noise_input_length = random.randint(2, 5) * 100  # 2 to 5 seconds
    noise_ma = np.zeros_like(lead_I)
    noise_em = np.zeros_like(lead_I)
    noise_bw = np.zeros_like(lead_I)

    noise_val_length = len(ma_val)
    st = random.randint(0, noise_val_length - noise_input_length)  # starting point from the noise signal for the em/ma
    st_bw = random.randint(0, noise_val_length - signal_length)
    st2_ma = random.randint(0, signal_length - noise_input_length)  # sample from the ecg where we will put the noise
    st2_em = random.randint(0, signal_length - noise_input_length)

    noise_input_ma = ma_val[st:st + noise_input_length]
    noise_input_em = em_val[st:st + noise_input_length]
    noise_input_bw = bw_val[st_bw:st_bw + signal_length]  # BW noise will affect the whole signal

    noise_ma[st2_ma:st2_ma + noise_input_length] = noise_input_ma
    noise_em[st2_em:st2_em + noise_input_length] = noise_input_em

    factor_bw = random.uniform(0.7, 1.3)
    factor_ma = random.uniform(0.7, 1.3)
    factor_em = random.uniform(1, 1.5)

    choice = random.randint(0, 2)  # 0 - ma; 1 - em; 2 - bw or any combination

    if choice == 0:  # ma
        noise_class = [1, 0, 0]
        noisy_ecg = leads[lead] + factor_ma * noise_ma
    elif choice == 1:  # em
        noise_class = [0, 1, 0]
        noisy_ecg = leads[lead] + factor_em * noise_em
    else:
        select_noise = ['bw', 'bw + ma', 'bw + em', 'ma + em', 'bw + ma + em']
        np.random.shuffle(select_noise)
        selection = select_noise[0]
        if selection == 'bw':
            noise_class = [0, 0, 1]
            noisy_ecg = leads[lead] + factor_bw * noise_input_bw
        elif selection == 'bw + ma':
            noise_class = [1, 0, 1]
            noisy_ecg = leads[lead] + factor_bw * noise_input_bw + factor_ma * noise_ma
        elif selection == 'bw + em':
            noise_class = [0, 1, 1]
            noisy_ecg = leads[lead] + factor_bw * noise_input_bw + factor_em * noise_em
        elif selection == 'ma + em':
            noise_class = [1, 1, 0]
            noisy_ecg = leads[lead] + factor_ma * noise_ma + factor_em * noise_em
        elif selection == 'bw + ma + em':
            noise_class = [1, 1, 1]
            noisy_ecg = leads[lead] + factor_bw * noise_input_bw + factor_ma * noise_ma + factor_em * noise_em

    with open(X_val_dir + str(i) + '.npy', 'wb') as f:
        np.save(f, noisy_ecg)

    with open(Y_class_val_dir + str(i) + '.npy', 'wb') as f:
        np.save(f, noise_class)


i = 9090
ecg_noi = np.load(X_train_dir + str(i) + '_I.npy')
ecg_clean = np.load(Y_train_dir + str(i) + '_I.npy')
clas = np.load(Y_class_train_dir + str(i) + '_I.npy')

plt.figure()
plt.plot(ecg_noi)
#plt.figure()
plt.plot(ecg_clean)
print(clas)


i = 5702
ecg_noi = np.load(X_train_dir + str(i) + '_I.npy')
ecg_clean = np.load(Y_train_dir + str(i) + '_I.npy')
clas = np.load(Y_class_train_dir + str(i) + '_I.npy')

plt.figure()
plt.plot(ecg_noi)
#plt.figure()
plt.plot(ecg_clean)
print(clas)

o=0
for file in os.listdir('data/Y/Y_train'):
    y = pp.minmax_scale(np.load('data/Y/Y_train/' + str(file)).reshape(1000, 1))  # minmaxscale - experiencia!
    # y = np.load(str(path_y) + '/' + str(file) + '.npy').reshape(1000, 1)
    X = pp.minmax_scale(np.load('data/X/X_train/' + str(file)).reshape(1000, 1))
    print(o)
    o=o+1
    if np.round(max(y)[0]) != 1:
        print('op1')
        print(file)
        print('max: '+str(int(max(y))))
        print('min: ' + str(int(min(y))))
    if np.round(min(y)[0]) != 0:
        print('op2')
        print(file)
        print('min: '+str(int(min(y))))
        print('max: ' + str(int(max(y))))
    if np.round(max(X)[0]) != 1:
        print('op3')
        print(file)
        print('max '+str(int(max(X))))
        print('min: ' + str(int(min(X))))
    if np.round(min(X)[0]) != 0:
        print('op4')
        print(file)
        print('min '+str(int(min(X))))
        print('max ' + str(int(max(X))))

