import numpy as np
import pandas as pd
import wfdb


def load_raw_data(bd, pn_dir, sr=500, path=None):
    records_list = pd.read_csv('data/records/' + str(bd), sep='/n', header=None, names=['filename'])
    data = []
    for f in records_list.filename:
        if sr == 100:
            file_name = f[-8:]
        elif sr == 500:
            file_name = f[-8:-2] + 'hr'
        if sr == 100:
            rest_path = f[:-8]
        elif sr == 500:
            rest_path = 'records500' + f[10:-8]
        content, meta = wfdb.rdsamp(str(file_name), pn_dir=str(pn_dir) + '/' + str(rest_path))
        data.append(content)
    return data


# as an alternative to loading the files directly from the web was not working, it is possible to download the zip file
# from physionet and load as npy files using the "wfdb.rdsamp" function
def load_raw_data_local(bd, local_dir, sr=500, path=None):
    records_list = pd.read_csv('data/records/' + str(bd), sep='/n', header=None, names=['filename'])
    rec = 'records' + str(sr)
    data = []
    for f in records_list.filename:
        if f[:10] == rec:
            content, meta = wfdb.rdsamp(str(local_dir) + '/' + str(f))
            data.append(content)
    return data


def ptbxl_save(data, save_dir='data500'):
    for i in range(np.shape(data)[0]):
        np.save(save_dir + '/' + str(i) + '.npy', data[i])

