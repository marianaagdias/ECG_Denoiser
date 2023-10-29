import random

import numpy as np
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt
import math
import sklearn.preprocessing as pp

import torch
from torch.utils.data import Dataset

import os


def configure_device(gpu_id):
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)


def configure_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def plot(epochs, plottable, ylabel='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')


def plot_losses(epochs, valid_losses, train_losses, ylabel ='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    #plt.xticks(epochs)
    plt.plot(epochs, valid_losses, label='validation')
    plt.plot(epochs, train_losses, label='train')
    plt.legend()
    plt.savefig('results/' + '%s.pdf' % (name), bbox_inches='tight')


class DatasetSequence(Dataset):
    '''
    path/labels_train
        /X_train
        /labels_val
        /X_val
        /labels_test
        /X_test
    '''
    def __init__(self, path, train_dev_test, part='train', features=1, overlap=None):
        self.path = path
        self.part = part
        self.train_dev_test = train_dev_test
        self.features = features
        self.overlap = overlap

    def __len__(self):
        if self.part == 'train':
            return self.train_dev_test[0]
        elif self.part == 'val':
            return self.train_dev_test[1]
        elif self.part in ['test', 'test2']:
            return self.train_dev_test[2]

    def __getitem__(self, idx):
        X, y = read_data(self.path, self.part, idx, self.features, self.overlap)
        # print(y.shape)
        return torch.tensor(X.copy()).float(), torch.tensor(y.copy()).float()


def read_data(path, partition, idx, features, overlap):
    path_y = str(path) + 'Y/Y_' + str(partition)
    path_X = str(path) + 'X/X_' + str(partition)
    if partition == 'train':
        if idx%3 == 0:
            file = str(math.floor(idx/3)) + '_1'
        elif idx%3 == 1:
            file = str(math.floor(idx/3)) + '_2'
        elif idx%3 == 2:
            file = str(math.floor(idx/3)) + '_3'
    else:
        file = idx
    signal = np.load(str(path_y) + '/' + str(file)+'.npy')
    signal_noisy = np.load(str(path_X) + '/' + str(file) + '.npy')
    signal = pp.minmax_scale(signal)
    signal_noisy = pp.minmax_scale(signal_noisy)
    if features == 1:
        timesteps = signal.shape[0]
        y = signal.reshape(timesteps, 1)
        X = signal_noisy.reshape(timesteps, 1)
    else:
        if overlap is None:
            timesteps = int(signal.shape[0]/features)
            y = signal.reshape(timesteps, features)
            X = signal_noisy.reshape(timesteps, features)
        else:
            len_ = signal.shape[0]-overlap
            step_size = features - overlap
            timesteps = int(len_/step_size)
            y = as_strided(signal, (timesteps, features), strides=(step_size*4, 4))
            X = as_strided(signal_noisy, (timesteps, features), strides=(step_size*4, 4))
    return X, y


def compute_class_scores(y_true, y_pred, matrix):
    for j in range(len(y_true)):
        pred = y_pred[j]
        gt = y_true[j]
        for i in range(0, 4): #for each class
            matrix = computetpfnfp(pred[i], gt[i], i, matrix)
    return matrix


def computetpfnfp(pred, gt, i, matrix):
    if gt==0 and pred==0: #tn
        matrix[i,3] +=1
    if gt==1 and pred==0: #fn
        matrix[i,1] +=1
    if gt==0 and pred==1: #fp
        matrix[i,2] +=1
    if gt==1 and pred==1: #tp
        matrix[i,0] +=1
    return matrix

