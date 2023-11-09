import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import os, sys, inspect
import scipy.signal as si
import scipy.stats as stats
import math
from tools.compute_metrics_with_GT import signaltonoise
import sklearn.preprocessing as pp
import tol_colors as tc


# access the noise data
pickle_in = open('data_noise.pickle', 'rb')
noise_in = pickle.load(pickle_in)


ma = np.concatenate((np.array(noise_in[-1][:, 0]), np.array(noise_in[-1][:, 1])))
em = np.concatenate((np.array(noise_in[-2][:, 0]), np.array(noise_in[-2][:, 1])))
bw = np.concatenate((np.array(noise_in[-3][:, 0]), np.array(noise_in[-3][:, 1])))

sample_rate = len(bw)/(30*2*60)  # the noise signals are 30 min long
new_s_rate = 360
num_samples = 360 * 30 * 60 * 2  # 360 Hz (ecg sampling rate)
ma = stats.zscore(si.resample(ma, num_samples))
em = stats.zscore(si.resample(em, num_samples))
bw = stats.zscore(si.resample(bw, num_samples))

noise_length = len(bw)  # 1296000 (30 min)

ma_train = ma[:int(noise_length*0.8)]
ma_val = ma[int(noise_length*0.8) : int(noise_length*0.9)]
ma_test = ma[int(noise_length*0.9):]

em_train = em[:int(noise_length*0.8)]
em_val = em[int(noise_length*0.8) : int(noise_length*0.9)]
em_test = em[int(noise_length*0.9):]

bw_train = bw[:int(noise_length*0.8)]
bw_val = bw[int(noise_length*0.8) : int(noise_length*0.9)]
bw_test = bw[int(noise_length*0.9):]


# ecg data - there was a problem with the ecg data folder from NAS so i downloaded the data
ecg_train_directory = 'data/Y_train_all_leads/'
ecg_val_directory = 'data/Y_val_all_leads/'
ecg_test_directory = 'data/Y_test_all_leads/'

n_train = 15259
n_val = 3270
n_test = 3267

Y_train_dir = 'data/Y/Y_train/'
Y_test_dir = 'data/Y/Y_test/'
Y_val_dir = 'data/Y/Y_val/'

Y_class_train_dir = 'data/Y_class/Y_train/'
Y_class_test_dir = 'data/Y_class/Y_test/'
Y_class_val_dir = 'data/Y_class/Y_val/'

X_train_dir = 'data/X/X_train/'
X_test_dir = 'data/X/X_test/'
X_val_dir = 'data/X/X_val/'


# Empty Datasets directories (in the case they contain anything)
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

# CREATE TRAIN SETS

for i in range(0, n_train):
    ecg = np.load(ecg_train_directory + str(i) + '.npy')

    peaks = []
    ord = []
    for j in range(0, 12):
        num_peaks = len(si.find_peaks(ecg[:, j])[0])
        peaks.append(num_peaks)
        ord.append(j)
    leads = [ord for _,ord in sorted(zip(peaks, ord))]

    # selection of the 3 leads with less peaks
    lead_1 = stats.zscore(ecg[:, leads[0]])
    lead_2 = stats.zscore(ecg[:, leads[1]])
    lead_3 = stats.zscore(ecg[:, leads[2]])

    leads = [lead_1, lead_2, lead_3]
    leads_names = ['1', '2', '3']
    signal_length = len(lead_1)

    # save each selected lead as an Y_train sample
    with open(Y_train_dir + str(i) + '_1' + '.npy', 'wb') as f:
        np.save(f, lead_1)
    with open(Y_train_dir + str(i) + '_2' + '.npy', 'wb') as f:
        np.save(f, lead_2)
    with open(Y_train_dir + str(i) + '_3' + '.npy', 'wb') as f:
        np.save(f, lead_3)

    # add noise to each signal and save as an X_train sample

    # create an array with same shape as the ecg, mostly composed by 0's and with the noise signal (2-6 s) in a random
    # position
    noise_ma = np.zeros_like(lead_1)
    noise_em = np.zeros_like(lead_1)
    noise_bw = np.zeros_like(lead_1)

    noise_input_length = random.randint(2, 6) * 360  # we are putting 2 to 6 seconds of noise in the signal

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

    with open(X_train_dir + str(i) + '_' + str(leads_names[leads_n[0]]) + '.npy', 'wb') as f:
        np.save(f, noisy_ecg_ma)

    with open(Y_class_train_dir + str(i) + '_' + str(leads_names[leads_n[0]]) + '.npy', 'wb') as f:
        np.save(f, noise_class_ma)

    # em noise
    start_noise = random.randint(0, signal_length - noise_input_length)
    noise_em[start_noise:start_noise + noise_input_length] = noise_input_em
    factor = random.uniform(1, 1.5)

    noisy_ecg_em = leads[leads_n[1]] + factor * noise_em
    noise_class_em = [0, 1, 0]

    with open(X_train_dir + str(i) + '_' + str(leads_names[leads_n[1]]) + '.npy', 'wb') as f:
        np.save(f, noisy_ecg_em)

    with open(Y_class_train_dir + str(i) + '_' + str(leads_names[leads_n[1]]) + '.npy', 'wb') as f:
        np.save(f, noise_class_em)

    # bw noise
    # for this sample, we will add bw noise OR bw and one of the other types of noise OR bw and both of the others

    select_noise = ['bw', 'bw + ma', 'bw + em', 'ma + em', 'bw + ma + em']
    np.random.shuffle(select_noise)
    selection = select_noise[0]

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

    with open(X_train_dir + str(i) + '_' + str(leads_names[leads_n[2]]) + '.npy', 'wb') as f:
        np.save(f, noisy_ecg_bw)

    with open(Y_class_train_dir + str(i) + '_' + str(leads_names[leads_n[2]]) + '.npy', 'wb') as f:
        np.save(f, noise_class_bw)

# plots
i = 90
ecg = np.load(ecg_train_directory + str(i) + '.npy')

peaks = []
ord = []
for j in range(0, 12):
    num_peaks = len(si.find_peaks(ecg[:, j])[0])
    peaks.append(num_peaks)
    ord.append(j)
leads = [ord for _,ord in sorted(zip(peaks, ord))]

# selection of the 3 leads with less peaks
lead_1 = stats.zscore(ecg[:, leads[0]])

signal_length = len(lead_1)

noise_ma = np.zeros_like(lead_1)
noise_input_length = random.randint(2, 6) * 360  # we are putting 2 to 6 seconds of noise in the signal

noise_train_length = len(ma_train)
st = random.randint(0, noise_train_length - noise_input_length)
noise_input_ma = ma_train[st:st + noise_input_length]

# ma noise
start_noise = random.randint(0, signal_length - noise_input_length)
noise_ma[start_noise:start_noise + noise_input_length] = noise_input_ma
factor = random.uniform(0.7, 1.3)

noisy_ecg_ma = pp.minmax_scale(lead_1*(-1)) + 1.3 * pp.minmax_scale(noise_ma)

noisy_ecg_em = pp.minmax_scale(lead_1*(-1)) + 1.3 * pp.minmax_scale(noise_em)
noisy_ecg_bw = pp.minmax_scale(lead_1*(-1)) + 1.3 * pp.minmax_scale(noise_input_bw)

x_label = np.linspace(0, 10, num=11, retstep=True)
x_label_noi = np.linspace(0, 40, num=11, retstep=True)
colors = list(tc.tol_cset('bright'))
plt.figure()
plt.plot(pp.minmax_scale(lead_1*(-1)), c=colors[-1])
plt.xticks(np.arange(0, 3601, step=360), x_label[0], fontsize=26)
plt.yticks(fontsize=26)
plt.xlabel('seconds', fontsize=28)
plt.ylabel('Amplitude (a.u.)', fontsize=28)
plt.figure()
plt.plot(pp.minmax_scale(ma_train), c=colors[-1])
plt.xticks(np.arange(0, len(em_train)+1, step=94255), x_label_noi[0], fontsize=26)
plt.yticks(fontsize=26)
plt.xlabel('minutes', fontsize=28)
plt.ylabel('Amplitude (a.u.)', fontsize=28)
plt.figure()
plt.plot(pp.minmax_scale(em_train), c=colors[-1])
plt.xticks(np.arange(0, len(em_train)+1, step=94255), x_label_noi[0], fontsize=26)
plt.yticks(fontsize=26)
plt.xlabel('minutes', fontsize=28)
plt.ylabel('Amplitude (a.u.)', fontsize=28)
plt.figure()
plt.plot(pp.minmax_scale(bw_train), c=colors[-1])
plt.xticks(np.arange(0, len(em_train)+1, step=94255), x_label_noi[0], fontsize=26)
plt.yticks(fontsize=26)
plt.xlabel('minutes', fontsize=28)
plt.ylabel('Amplitude (a.u.)', fontsize=28)
plt.figure()
plt.plot(noisy_ecg_ma, c=colors[-1])
plt.xticks(np.arange(0, 3601, step=360), x_label[0], fontsize=26)
plt.yticks(fontsize=26)
plt.xlabel('seconds', fontsize=28)
plt.ylabel('Amplitude (a.u.)', fontsize=28)

plt.figure()
plt.plot(noisy_ecg_em, c=colors[-1])
plt.xticks(np.arange(0, 3601, step=360), x_label[0], fontsize=26)
plt.yticks(fontsize=26)
plt.xlabel('seconds', fontsize=28)
plt.ylabel('Amplitude (a.u.)', fontsize=28)

plt.figure()
plt.plot(noisy_ecg_bw, c=colors[-1])
plt.xticks(np.arange(0, 3601, step=360), x_label[0], fontsize=26)
plt.yticks(fontsize=26)
plt.xlabel('seconds', fontsize=28)
plt.ylabel('Amplitude (a.u.)', fontsize=28)


# CREATE VALIDATION SETS
for i in range(0, n_val):
    ecg_all = np.load(ecg_val_directory + str(i) + '.npy')

    peaks = []
    ord = []
    for j in range(0, 12):
        num_peaks = len(si.find_peaks(ecg_all[:, j])[0])
        peaks.append(num_peaks)
        ord.append(j)
    leads = [ord for _, ord in sorted(zip(peaks, ord))]

    # instead of selecting a random lead, we will select one of the 3 leads with less peaks (in order to avoid having
    # very noisy "clean" signals in the validation set.
    lead = random.randint(0, 2)
    ecg = ecg_all[:, leads[lead]]
    signal_length = len(ecg)

    # save the lead as an Y_train sample
    with open(Y_val_dir + str(i) + '.npy', 'wb') as f:
        np.save(f, ecg)

    # add noise to each signal and save as an X_train sample

    # create an array with same shape as the ecg, mostly composed by 0's and with the noise signal (4 s) in a random
    # position
    noise_input_length = random.randint(2, 6) * 360  # 2 to 6 seconds
    noise_ma = np.zeros_like(ecg)
    noise_em = np.zeros_like(ecg)
    noise_bw = np.zeros_like(ecg)

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
        noisy_ecg = ecg + factor_ma * noise_ma
    elif choice == 1:  # em
        noise_class = [0, 1, 0]
        noisy_ecg = ecg + factor_em * noise_em
    else:
        select_noise = ['bw', 'bw + ma', 'bw + em', 'ma + em', 'bw + ma + em']
        np.random.shuffle(select_noise)
        selection = select_noise[0]
        if selection == 'bw':
            noise_class = [0, 0, 1]
            noisy_ecg = ecg + factor_bw * noise_input_bw
        elif selection == 'bw + ma':
            noise_class = [1, 0, 1]
            noisy_ecg = ecg + factor_bw * noise_input_bw + factor_ma * noise_ma
        elif selection == 'bw + em':
            noise_class = [0, 1, 1]
            noisy_ecg = ecg + factor_bw * noise_input_bw + factor_em * noise_em
        elif selection == 'ma + em':
            noise_class = [1, 1, 0]
            noisy_ecg = ecg + factor_ma * noise_ma + factor_em * noise_em
        elif selection == 'bw + ma + em':
            noise_class = [1, 1, 1]
            noisy_ecg = ecg + factor_bw * noise_input_bw + factor_ma * noise_ma + factor_em * noise_em

    with open(X_val_dir + str(i) + '.npy', 'wb') as f:
        np.save(f, noisy_ecg)

    with open(Y_class_val_dir + str(i) + '.npy', 'wb') as f:
        np.save(f, noise_class)



# CREATE TEST SETS (2 - used for the paper) - Adding noise of known SNRin values of either 0, 5, 7, or 10 dB

for file in os.listdir('data/Y_class/Y_test2'):
    os.remove(str('data/Y_class/Y_test2/' + file))
for file in os.listdir('data/Y/Y_test2'):
    os.remove(str('data/Y/Y_test2/' + file))
for file in os.listdir('data/X/X_test2'):
    os.remove(str('data/X/X_test2/' + file))


def compute_factor_Xdb(ecg, noise, db=0):
    frac = sum(ecg**2) / sum(noise**2)

    fac = frac/(10**(db/10))

    # noise_to_add = noise*math.sqrt(fac)

    # noisy_ecg = ecg + noise_to_add

    return math.sqrt(fac)


# CREATE TEST SETS 2 ----- saving the timesteps  choosing the leads
for i in range(0, n_test):
    ecg_all = np.load('data/Y_test_all_leads/' + str(i) + '.npy')

    peaks = []
    ord = []
    for j in range(0, 12):
        num_peaks = len(si.find_peaks(ecg_all[:, j])[0])
        peaks.append(num_peaks)
        ord.append(j)
    leads = [ord for _, ord in sorted(zip(peaks, ord))]

    # instead of selecting a random lead, we will select one of the 3 leads with less peaks (in order to avoid having
    # very noisy "clean" signals in the validation set.
    lead = random.randint(0, 2)
    ecg = pp.minmax_scale(ecg_all[:, leads[lead]])
    signal_length = len(ecg)

    # save the lead as an Y_test sample
    with open('data/Y/Y_test2/' + str(i) + '.npy', 'wb') as f:
        np.save(f, ecg)

    # add noise to each signal and save as an X_train sample

    # create an array with same shape as the ecg, mostly composed by 0's and with the noise signal (4 s) in a random
    # position
    noise_input_length = random.randint(2, 6) * 360  # 2 to 6 seconds
    noise_ma = np.zeros_like(ecg)
    noise_em = np.zeros_like(ecg)
    noise_bw = np.zeros_like(ecg)

    noise_test_length = len(ma_test)
    st = random.randint(0, noise_test_length - noise_input_length)  # starting point from the noise signal for the em/ma
    st_bw = random.randint(0, noise_test_length - signal_length)
    st2 = random.randint(0, signal_length - noise_input_length)  # sample from the ecg where we will put the noise
    # st2_em = random.randint(0, signal_length - noise_input_length)

    noise_input_ma = pp.minmax_scale(ma_test[st:st + noise_input_length])
    noise_input_em = pp.minmax_scale(em_test[st:st + noise_input_length])
    noise_input_bw = pp.minmax_scale(bw_test[st_bw:st_bw + signal_length])  # BW noise will affect the whole signal

    noise_ma[st2:st2 + noise_input_length] = noise_input_ma
    noise_em[st2:st2 + noise_input_length] = noise_input_em

    snr_in_all = [0, 5, 7, 10]
    snr_in = snr_in_all[random.randint(0, 3)]

    # save the start and ending points of the noise AND the snr_in
    ts_st_end = [st2, st2 + noise_input_length]
    info = np.append(ts_st_end, snr_in)
    with open('data/Y_test_noise_timesteps/' + str(i) + '.npy', 'wb') as f:
        np.save(f, info)

    choice = random.randint(0, 2)  # 0 - ma; 1 - em; 2 - bw or any combination

    if choice == 0:  # ma
        noise_class = [1, 0, 0]
        fact = compute_factor_Xdb(ecg[st2:st2+noise_input_length], noise_ma[st2:st2+noise_input_length], db=snr_in)
        noisy_ecg = ecg + fact * noise_ma
        print(signaltonoise(ecg[st2:st2+noise_input_length], noise_ma[st2:st2+noise_input_length]))
    elif choice == 1:  # em
        noise_class = [0, 1, 0]
        fact = compute_factor_Xdb(ecg[st2:st2 + noise_input_length], noise_em[st2:st2 + noise_input_length], db=snr_in)
        noisy_ecg = ecg + fact * noise_em
    else:
        select_noise = ['bw', 'bw + ma', 'bw + em', 'ma + em', 'bw + ma + em']
        np.random.shuffle(select_noise)
        selection = select_noise[0]
        if selection == 'bw':
            noise_class = [0, 0, 1]
            fact = compute_factor_Xdb(ecg, noise_input_bw, db=snr_in)
            noisy_ecg = ecg + fact * noise_input_bw
        elif selection == 'bw + ma':
            noise_class = [1, 0, 1]
            noise = noise_input_bw + noise_ma
            fact = compute_factor_Xdb(ecg, noise, db=snr_in)
            noisy_ecg = ecg + fact*noise
        elif selection == 'bw + em':
            noise_class = [0, 1, 1]
            noise = noise_input_bw + noise_em
            fact = compute_factor_Xdb(ecg, noise, db=snr_in)
            noisy_ecg = ecg + fact * noise
        elif selection == 'ma + em':
            noise_class = [1, 1, 0]
            noise = noise_em + noise_ma
            fact = compute_factor_Xdb(ecg[st2:st2 + noise_input_length], noise[st2:st2 + noise_input_length],
                                      db=snr_in)
            noisy_ecg = ecg + fact * noise
        elif selection == 'bw + ma + em':
            noise_class = [1, 1, 1]
            noise = noise_input_bw + noise_em + noise_ma
            fact = compute_factor_Xdb(ecg, noise, db=snr_in)
            noisy_ecg = ecg + fact * noise

    with open('data/X/X_test2/' + str(i) + '.npy', 'wb') as f:
        np.save(f, noisy_ecg)

    with open('data/Y_class/Y_test2/' + str(i) + '.npy', 'wb') as f:
        np.save(f, noise_class)


# check
i = 5
clean = np.load('data/Y/Y_test2/' + str(i) + '.npy')
noisy = np.load('data/X/X_test2/' + str(i) + '.npy')
info = np.load('data/Y_test_noise_timesteps/' + str(i) + '.npy')
classi = np.load('data/Y_class/Y_test2/' + str(i) + '.npy')

plt.figure()
plt.plot(clean)
plt.plot(noisy)
print(info)
print(classi)

print(signaltonoise(clean[info[0]: info[1]], noisy[info[0]: info[1]]))
print(signaltonoise(clean[info[0]: info[1]], pp.minmax_scale(noisy[info[0]: info[1]])))



# CREATE TEST SETS (not used for the paper)
for i in range(0, n_test):
    ecg_all = np.load(ecg_test_directory + str(i) + '.npy')
    # randomly select 1 ecg lead
    lead = random.randint(0, 11)
    ecg = ecg_all[:, lead]
    signal_length = len(ecg)

    # save the lead as an Y_train sample
    with open(Y_test_dir + str(i) + '.npy', 'wb') as f:
        np.save(f, ecg)

    # add noise to each signal and save as an X_train sample

    # create an array with same shape as the ecg, mostly composed by 0's and with the noise signal (4 s) in a random
    # position
    noise_input_length = random.randint(2, 6) * 360  # 2 to 6 seconds
    noise_ma = np.zeros_like(ecg)
    noise_em = np.zeros_like(ecg)
    noise_bw = np.zeros_like(ecg)

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
        noisy_ecg = ecg + factor_ma * noise_ma
    elif choice == 1:  # em
        noise_class = [0, 1, 0]
        noisy_ecg = ecg + factor_em * noise_em
    else:
        select_noise = ['bw', 'bw + ma', 'bw + em', 'ma + em', 'bw + ma + em']
        np.random.shuffle(select_noise)
        selection = select_noise[0]
        if selection == 'bw':
            noise_class = [0, 0, 1]
            noisy_ecg = ecg + factor_bw * noise_input_bw
        elif selection == 'bw + ma':
            noise_class = [1, 0, 1]
            noisy_ecg = ecg + factor_bw * noise_input_bw + factor_ma * noise_ma
        elif selection == 'bw + em':
            noise_class = [0, 1, 1]
            noisy_ecg = ecg + factor_bw * noise_input_bw + factor_em * noise_em
        elif selection == 'ma + em':
            noise_class = [1, 1, 0]
            noisy_ecg = ecg + factor_ma * noise_ma + factor_em * noise_em
        elif selection == 'bw + ma + em':
            noise_class = [1, 1, 1]
            noisy_ecg = ecg + factor_bw * noise_input_bw + factor_ma * noise_ma + factor_em * noise_em

    with open(X_test_dir + str(i) + '.npy', 'wb') as f:
        np.save(f, noisy_ecg)

    with open(Y_class_test_dir + str(i) + '.npy', 'wb') as f:
        np.save(f, noise_class)