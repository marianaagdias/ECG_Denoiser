# ECG Signal Noise Removal with Bidirectional GRU

This repository contains a PyTorch implementation of a gated recurrent unit (GRU) model for the task of removing noise from electrocardiogram (ECG) signals.

## Abstract

As the popularity of wearables continues to scale, a substantial portion of the population has now access to
(self-)monitorization of cardiovascular activity. In particular, the use of ECG wearables is growing in the realm of
occupational health assessment, but one common issue that is encountered is the presence of noise which hinders the
reliability of the acquired data. In this work, we propose an ECG denoiser based on bidirectional Gated Recurrent Units
(biGRU). This model was trained on noisy ECG samples that were created by adding noise from the MIT-BIH Noise Stress
Test database to ECG samples from the PTB-XL database. The model was initially trained and tested on data corrupted
with the three most common sources of noise: electrode motion artifacts, muscle activation and baseline wander. After
training, the model was able to fully reconstruct previously unseen signals, achieving Root-Mean-Square Error values
between 0.041 and 0.023. For further testing the model's robustness, we performed a data collection in an industrial
work setting and employed our model to clean the noisy data, acquired from 43 workers using wearable sensors. The
trained network proved to be very effective in removing real ECG noise, outperforming the available open-source
solutions, while having a much smaller complexity compared to state-of-the-art Deep Learning approaches.

## Features

- GRU-based model for time series analysis
- Bidirectional GRU for better context capturing
- Data preprocessing techniques implemented
- Trained model weights are provided for public usage

## Installation

```bash
git clone https://github.com/marianaagdias/ECG_Denoiser.git
cd ECG_Denoiser
pip install -r requirements.txt

```

## Databases
For the development of the ECG noise removal model, in the first place, we created a dataset with noisy and clean
versions of ECG signals. For that, we used the PTB-XL and the MIT-BIH Noise Stress Test databases. These are two 
publicly available datasets from Physionet and can be accessed in:
- PTB-XL: https://physionet.org/content/ptb-xl/1.0.3/
- MIT-BIH Noise Stress Test: https://physionet.org/content/nstdb/1.0.0/


## Code Structure

tools: package including the necessary functions for the project.

### Dataset Creation
In the "create_dataset" folder, it is possible to find 3 scripts that were used to create the train, validation and test
sets, more concretely:
- 1_get_save_data_from_db.py : script for loading the data from the databases
- 2_train_val_test_split: data curation; splitting the data in train, validation and test sets; pre-processing of the 
clean (ground truth) ECG data from the PTB-XL database
- 3_create_noisy_data2_360hz: creation of noisy ECG records (to be used as input of the model) from the clean ECG 
records and the noise data from the MIT-BIH noise stress test db.

### Model
- gru_denoiser.py and utils_denoiser.py: PyTorch implementation of a gated recurrent unit (GRU) model for the task of removing noise from electrocardiogram (ECG) signals
- evaluate_test_set.py: script for testing the trained model
- best_gru_denoiser_360Hz: weights of the trained model after performing hyperparameter optimization on the validation set


