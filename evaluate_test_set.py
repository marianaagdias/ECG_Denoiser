from torch.utils.data import DataLoader
from gru_denoiser import evaluate_test
import torch
from gru_denoiser import GRU
from utils_denoiser import DatasetSequence

gpu_id = None
device = 'cpu'

model = GRU(n_features=1, hid_dim=64, n_layers=1, dropout=0, gpu_id=None, bidirectional=True)
path_to_weights = 'best_gru_denoiser_360Hz'
model.load_state_dict(torch.load(path_to_weights, map_location=torch.device(device)))

# model in the evaluation mode
model.eval()

# test dataset
test_dataset = DatasetSequence('data_500/', [45777, 3270, 3267], 'test2')
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# evaluate the performance of the model
test_RMSE = evaluate_test(model, dataloader=test_dataloader, file_name='results_64_bi_drop0_test_set_db.txt', directory_res='results/y_pred_360/',
              features=1, gpu_id=None)


