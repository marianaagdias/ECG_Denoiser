import numpy as np
import pickle
from tools.load_physionet_data import load_raw_data, load_raw_data_local, ptbxl_save

# load from local folder
rec_file = 'RECORDS'
local_dir = 'data_raw/ptb_xl'
data = load_raw_data_local(rec_file, local_dir, sr=500)
data = np.array(data)
# save each record separately as a numpy array
dir = 'data/ptb_xl_500hz'
ptbxl_save(data, dir)

# load MIT-BIH noise database
rec_file_noise = 'records_noise'
pn_dir_noise = 'nstdb'
data_noise = load_raw_data(rec_file_noise, pn_dir_noise)
# save
pickle_out = open("data_noise.pickle", "wb")
pickle.dump(data_noise, pickle_out)
pickle_out.close()
