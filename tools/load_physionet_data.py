import numpy as np
import pandas as pd
import wfdb


def load_raw_data(rec_file, pn_dir, sr=500):
    """
    function that creates a variable with containing all records data from the PhysioNet database that can be found in
    https://physionet.org/content/{pn_dir}.
    PTB-XL database: https://physionet.org/content/ptb-xl/1.0.3/
    MIT-BIH Noise Stress Test Database: https://physionet.org/content/nstdb/1.0.0/
    :param rec_file: file directory where the records names from the database are listed. it should be downloaded from
    the physionet website (ex: https://physionet.org/content/nstdb/1.0.0/RECORDS)
    :param pn_dir: database acronym ('ptb-xl' or 'nstdb')
    :param sr: sampling rate of the data
    :return: list containing all records
    """
    records_list = pd.read_csv(rec_file, sep='/n', header=None, names=['filename'])
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


# as an alternative to loading the files directly from the web, it is possible to download the zip file
# from physionet and load as npy files using the "wfdb.rdsamp" function
def load_raw_data_local(rec_file, local_dir, sr=500):
    """
        function that creates a variable with containing all records data from the PhysioNet database downloaded to
        local_dir
        :param rec_file: file directory where the records names from the database are listed. it should be downloaded from
        the physionet website (ex: https://physionet.org/content/nstdb/1.0.0/RECORDS)
        :param local_dir: local directory containing the data
        :param sr: sampling rate of the data
        :return: list containing all records
    """
    records_list = pd.read_csv(rec_file, sep='/n', header=None, names=['filename'])
    rec = 'records' + str(sr)
    data = []
    for f in records_list.filename:
        if f[:10] == rec:
            content, meta = wfdb.rdsamp(str(local_dir) + '/' + str(f))
            data.append(content)
    return data


def ptbxl_save(data, save_dir='data500'):
    """
    function that saves each record from "data" as a separate numpy file
    :param data: array containing all records
    :param save_dir: directory where the numpy files should be saved
    :return: each record saved as a numpy file in save_dir
    """
    for i in range(np.shape(data)[0]):
        np.save(save_dir + '/' + str(i) + '.npy', data[i])

