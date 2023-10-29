import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
import pickle

import os, sys, inspect

from tools import pre_processing as pp

# pcs = ['my_pc', 'server']
# pc = pcs[0]

# project_dir = os.getcwd()

# if pc == 'my_pc':
#     webdav_dir = 'Z://DeepLearning/Data/ECG/PTB-XL/records100'
# else:
#     webdav_dir = '/run/user/1001/gvfs/dav:host=10.136.24.49,ssl=false,prefix=%2Fowncloud%2Fremote.php%2Fwebdav/' \
#                  'DeepLearning/Data/ECG/PTB-XL/records100'


def load_raw_data(bd, pn_dir, sr=100, path=None):
    records_list = pd.read_csv('data/records/' + str(bd), sep='/n', header=None, names=['filename'])
    data = []
    for f in records_list.filename:
        if sr == 100:
            file_name = f[-8:]  # ATENTIONNN - THIS WAS DONE BASED ON THE PTB-XL "records" file
        elif sr == 500:
            file_name = f[-8:-2] + 'hr'
        if sr == 100:
            rest_path = f[:-8]
        elif sr == 500:
            rest_path = 'records500' + f[10:-8]
        content, meta = wfdb.rdsamp(str(file_name), pn_dir=str(pn_dir) + '/' + str(rest_path))
        data.append(content)
    return data


# loading the files directly from the web was not working, so i downloaded the zip file from physionet and loaded as npy
# files using the "wfdb.rdsamp" function as well (february 2023)
def load_raw_data_local(bd, local_dir, sr=100, path=None):
    records_list = pd.read_csv('data/records/' + str(bd), sep='/n', header=None, names=['filename'])
    rec = 'records' + str(sr)
    data = []
    for f in records_list.filename:
        if f[:10] == rec:
            content, meta = wfdb.rdsamp(str(local_dir) + '/' + str(f))
            data.append(content)
    return data


# load PTB-XL
# bd = 'PTB-XL'
# pn_dir = 'ptb-xl'
# data = load_raw_data(bd, pn_dir, sr=500)  # load the files with 500 Hz sampling frequency

# load from local folder
bd = 'RECORDS'
local_dir = r'C:\Users\Catia Bastos\Downloads'
data = load_raw_data_local(bd, local_dir, sr=500)

# PRE-PROCESS (FILTER AND NORMALIZE) AND SAVE DATA
# change directory to the NAS folder

# os.chdir(webdav_dir)

data = np.array(data)


def ptbxl_save(data, save_dir='data500'):

    for i in range(np.shape(data)[0]):

        np.save(save_dir + '/' + str(i) + '.npy', data[i])

    return


# pp.ptbxl_preproc_save(data)
ptbxl_save(data)  # a aplicação do filtro passou a ser feita no "2_train_val_test_split"


# save the data
# pickle_out = open("data.pickle", "wb")
# pickle.dump(data, pickle_out)
# pickle_out.close()


# load MIT-BIH noise database

# os.chdir(project_dir)

bd_noise = 'mit-bih-noise'
pn_dir_noise = 'nstdb'
data_noise = load_raw_data(bd_noise, pn_dir_noise)

# MIT-BIH-noise
# change directory
if pc == 'my_pc':
    webdav_dir = 'Z://DeepLearning/Data/ECG/MIT-BIH-noise'
else:
    webdav_dir = '/run/user/1001/gvfs/dav:host=10.136.24.49,ssl=false,prefix=%2Fowncloud%2Fremote.php%2Fwebdav/' \
                 'DeepLearning/Data/ECG/MIT-BIH-noise'

os.chdir(webdav_dir)

# improve this part - what is the best form to keep the data?
pickle_out = open("data_noise.pickle", "wb")
pickle.dump(data_noise, pickle_out)
pickle_out.close()
