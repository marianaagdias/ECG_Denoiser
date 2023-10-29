import pandas as pd
import numpy as np
import os


def read_ae_data(data_dir):
    m = pd.read_csv(data_dir, header=3, delimiter='\t', names=['n', 'di', 'ECG', 'RIP Chest', 'RIP Abd', 'X', 'Y', 'Z'])
    return m


def read_sub_ecg(subject, ae_dir='factory_data', i=0, f=None):
    if subject < 10:
        path = ae_dir + r'\S0' + str(subject) + r'\rec'
    else:
        path = ae_dir + r'\S' + str(subject) + r'\rec'

    # check the files inside the folder "path" and select the "hub data"
    files = os.listdir(path)
    file = ''
    for fi in files:
        if 'hub' in fi:
            file = fi
    file_dir = path + r'\\' + file

    hub_data = read_ae_data(file_dir)

    if subject == 42:  # this subject's ECG is inverted (the position of the eletrodes was inverted by mistake)
        if f:
            ecg_sig = (-1) * np.array(hub_data['ECG'])[i:f]
        else:
            ecg_sig = (-1) * np.array(hub_data['ECG'])[i:]
    else:
        if f:
            ecg_sig = np.array(hub_data['ECG'])[i:f]
        else:
            ecg_sig = np.array(hub_data['ECG'])[i:]

    return ecg_sig

