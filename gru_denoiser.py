import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as f
from utils_denoiser import configure_seed, configure_device, plot, DatasetSequence
from datetime import datetime
import numpy as np
import pandas as pd
import os


class GRU(nn.Module):
    def __init__(self, n_features, hid_dim, n_layers, dropout, gpu_id=None,
                 bidirectional=False, attention=False, **kwargs):

        super(GRU, self).__init__()

        self.n_features = n_features
        self.n_layers = n_layers
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.gpu_id = gpu_id
        self.bidirectional = bidirectional

        # X - > (seq_length, batch_size, n_features) (input of the model)
        # in this case, n_features = 1

        self.gru_e1 = nn.GRU(input_size=n_features, hidden_size=hid_dim, num_layers=self.n_layers,  # dropout=dropout,
                               bidirectional=bidirectional, batch_first=True)
        self.drop = nn.Dropout(p=dropout)
        if self.bidirectional:
            self.d = 2
        else:
            self.d = 1
        self.gru_e2 = nn.GRU(input_size=hid_dim*self.d, hidden_size=n_features, num_layers=1, batch_first=True)

    def forward(self, x):
        # x - > (batch_size, seq_length, n_features) - source ecg

        # initialize hidden state:
        # h_0 = torch.zeros(self.d*self.n_layers, x.size(dim=0), self.hid_dim).to(self.gpu_id)

        x1, _ = self.gru_e1(x.to(self.gpu_id))  # ,(h_0))
        x1 = self.drop(x1)
        xe, _ = self.gru_e2(x1)

        # x1 shape: (batch_size, seq_length, d*hid_dim) # batch_first True
        # h_1 -> (d*num_layers, batch_size, hid_dim)
        # xe -> (batch_size, seq_length, d*n_features)

        return xe


def train_batch(X, Y, model, optimizer, criterion, gpu_id=None):
    X, Y = X.to(gpu_id), Y.to(gpu_id)

    model.train()
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, Y)
    loss.backward()
    optimizer.step()

    return loss.item()


def compute_val_loss(model, dataloader, gpu_id=None):
    model.eval()  # set dropout and batch normalization layers to evaluation
    with torch.no_grad():
        loss_batch = []
        for i, (x_batch, y_batch) in enumerate(dataloader):
            print('eval {} of {}'.format(i + 1, len(dataloader)), end='\r')
            x_batch, y_batch = x_batch.to(gpu_id), y_batch.to(gpu_id)

            y_pred = model(x_batch)
            loss_batch.append(f.mse_loss(y_pred, y_batch, reduction='mean'))

            del x_batch
            del y_batch
            torch.cuda.empty_cache()

        model.train()

        return torch.mean(torch.tensor(loss_batch).float()).item()


def evaluate_test(model, dataloader, file_name, directory_res, features, gpu_id=None):
    model.eval()  # set dropout and batch normalization layers to evaluation
    with torch.no_grad():
        loss_batch = []
        for i, (x_batch, y_batch) in enumerate(dataloader):
            print('eval {} of {}'.format(i + 1, len(dataloader)), end='\r')
            x_batch, y_batch = x_batch.to(gpu_id), y_batch.to(gpu_id)

            y_pred = model(x_batch)

            loss_batch.append(f.mse_loss(y_pred, y_batch, reduction='mean'))

            j = 0
            for sample1 in y_batch:
                if features == 1:
                    x_noisy = pd.Series(np.array(x_batch.cpu()[j])[:, 0], name='Noisy')
                    y_real = pd.Series(np.array(sample1.cpu())[:, 0], name='Real')
                    y_pred_ = pd.Series(np.array(y_pred.cpu()[j])[:, 0], name='Predicted')
                    y_df = pd.concat([x_noisy, y_real, y_pred_], axis=1)
                    y_df.to_csv(str(directory_res) + str(file_name) + '.txt', mode='a')
                else:
                    x_noisy = pd.Series(np.array(x_batch.cpu()[j].flatten(), dtype=float), name='Noisy')
                    y_real = pd.Series(np.array(sample1.cpu(), dtype=float).flatten(), name='Real')
                    y_pred_ = pd.Series(np.array(y_pred.cpu()[j], dtype=float).flatten(), name='Predicted')
                    y_df = pd.concat([x_noisy, y_real, y_pred_], axis=1)
                    y_df.to_csv(str(directory_res) + str(file_name) + '.txt', mode='a')
                j = j+1

            del x_batch
            del y_batch
            torch.cuda.empty_cache()

        model.train()

        return torch.mean(torch.tensor(loss_batch).float()).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', default='data/',
                        help="Path to the dataset.")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train the model.""")
    parser.add_argument('-batch_size', default=256, type=int,
                        help="Size of training batch.")
    parser.add_argument('-learning_rate', type=float, default=0.005)
    parser.add_argument('-dropout', type=float, default=0.3)
    parser.add_argument('-l2_decay', type=float, default=0)
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='adam')
    parser.add_argument('-gpu_id', type=int, default=None)
    parser.add_argument('-path_results', default='results/',
                        help='Path to save the model')
    parser.add_argument('-hidden_size', type=int, default=256)
    parser.add_argument('-n_layers', type=int, default=1)
    parser.add_argument('-bidirectional', type=bool, default=False)
    parser.add_argument('-attention', type=bool, default=False)
    parser.add_argument('-num_features', type=int, default=1)
    parser.add_argument('-overlap', type=int, default=None)
    parser.add_argument('-frequency', type=int, default=360)
    opt = parser.parse_args()

    directory_results = opt.path_results
    dir_save_models = directory_results + 'save_models_' + str(opt.frequency) + '/'
    dir_loss_file = directory_results + 'loss_' + str(opt.frequency) + '/'
    dir_plots = directory_results + 'plots_' + str(opt.frequency) + '/'
    dir_y_pred = directory_results + 'y_pred_' + str(opt.frequency) + '/'

    date = datetime.timestamp(datetime.now())
    results_file = 'gru_denoi_freq'+ str(opt.frequency) + '_' + str(opt.n_layers) + 'layers_' + 'hidden' + \
                   str(opt.hidden_size) + '_dropout' + str(opt.dropout) + '_' + str(opt.num_features) + 'feat' + '_' +\
                   'overl' + str(opt.overlap) + '_' + 'maxepoch' + str(opt.epochs) + 'bi' + str(opt.bidirectional) +\
                   '_' + str(date)

    configure_seed(seed=42)
    configure_device(opt.gpu_id)

    # model's parameters
    emb_dim = 32
    n_features = opt.num_features
    hid_dim = opt.hidden_size
    n_layers = opt.n_layers
    dropout = opt.dropout

    samples = [45777, 3270, 3267]
    print("Loading data...")
    train_dataset = DatasetSequence(opt.data, samples, 'train', n_features, overlap=opt.overlap)
    dev_dataset = DatasetSequence(opt.data, samples, 'val', n_features, overlap=opt.overlap)
    test_dataset = DatasetSequence(opt.data, samples, 'test', n_features, overlap=opt.overlap)

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = opt.gpu_id

    # initialize the model
    model = GRU(n_features=n_features, emb_dim=emb_dim, hid_dim=hid_dim, dropout=dropout,
                    bidirectional=opt.bidirectional, n_layers=n_layers, gpu_id=device).to(device)

    # get an optimizer
    optims = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD}

    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(),
        lr=opt.learning_rate,
        weight_decay=opt.l2_decay)

    criterion = nn.MSELoss(reduction='mean')

    # training loop
    epochs = torch.arange(1, opt.epochs + 1)
    valid_loss = []
    train_losses = []
    for ii in epochs:
        print('Training epoch {}'.format(ii))
        for i, (X_batch, y_batch) in enumerate(train_dataloader):
            # print(X_batch.shape, y_batch.shape)
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion, gpu_id=opt.gpu_id)
            del X_batch
            del y_batch
            torch.cuda.empty_cache()
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % mean_loss)

        train_losses.append(mean_loss)
        val_loss = compute_val_loss(model, dev_dataloader, gpu_id=device)
        valid_loss.append(val_loss)
        print('Validation Loss: %.4f' % (valid_loss[-1]))

        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        # save the model at each epoch where the validation loss is the best so far
        if val_loss == np.min(valid_loss):
            directory = os.path.join(dir_save_models, str(results_file) + '_model' + str(ii.item()))
            best_model = ii
            torch.save(model.state_dict(), directory)

    # Make predictions based on best model (lowest validation loss)
    # Load model
    model.load_state_dict(torch.load(directory))
    model.eval()

    # Results on test set:
    test_loss = evaluate_test(model, test_dataloader, gpu_id=device, file_name=results_file, directory_res=dir_y_pred,
                              features=n_features)
    print('Final Test Results: ' + str(test_loss))

    nam = pd.Series(results_file)
    res = pd.Series(test_loss)
    val_loss_final = pd.Series(np.min(valid_loss))
    res_df = pd.concat([nam, res, val_loss_final], axis=1)
    res_df.to_csv(str(dir_loss_file) + 'loss_results.txt', mode='a')

    # plot
    plot(epochs, valid_loss, ylabel='Validation Loss', name=str(dir_plots) + str(results_file))


if __name__ == '__main__':
    main()
