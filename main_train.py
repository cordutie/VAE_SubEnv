import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from modules.dataset import *
from modules.seeds import *
from modules.architecture import *
from modules.trainer import *

# Model 1 -------------------------------------------------------------------------------------

audio_path = 'sounds/fire_clean_augmented.wav'

frame_size, N_filter_bank, param_per_env = 17640, 24, 512
hidden_size, deepness = 1024, 2
latent_dim = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sr = 44100
seed = seed_maker(frame_size, sr, N_filter_bank)
seed = seed.to(device)

#Dataset loading
dataset = VAE_Dataset(audio_path, frame_size, sr)
dataset_list = dataset.compute_dataset()
dataloader = DataLoader(dataset_list, batch_size=64, shuffle=True)

#Model loading
modelo = VAE_SubEnv(frame_size, hidden_size, deepness, latent_dim, N_filter_bank, param_per_env, seed, device)

#Optimizer
optimizer = optim.Adam(modelo.parameters(), lr=1e-3)

#Training
model_path = 'models/multiscale/fire/'
train_multiscale_loss(modelo, optimizer, 100, device, dataloader, model_path)
model_path = 'models/statistics/fire/'
train_statistics_loss(modelo, optimizer, 100, device, dataloader, model_path, N_filter_bank, frame_size, sr)

# Model 2 -------------------------------------------------------------------------------------

audio_path = 'sounds/fire_clean_augmented.wav'

frame_size, N_filter_bank, param_per_env = 17640, 24, 512
hidden_size, deepness = 2048, 3
latent_dim = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sr = 44100
seed = seed_maker(frame_size, sr, N_filter_bank)
seed = seed.to(device)

#Dataset loading
dataset = VAE_Dataset(audio_path, frame_size, sr)
dataset_list = dataset.compute_dataset()
dataloader = DataLoader(dataset_list, batch_size=64, shuffle=True)

#Model loading
modelo = VAE_SubEnv(frame_size, hidden_size, deepness, latent_dim, N_filter_bank, param_per_env, seed, device)

#Optimizer
optimizer = optim.Adam(modelo.parameters(), lr=1e-3)

#Training
model_path = 'models/multiscale/fire_deep/'
train_multiscale_loss(modelo, optimizer, 100, device, dataloader, model_path)
model_path = 'models/statistics/fire_deep/'
train_statistics_loss(modelo, optimizer, 100, device, dataloader, model_path)

# Model 3 -------------------------------------------------------------------------------------

audio_path = 'sounds/water_augmented.wav'

frame_size, N_filter_bank, param_per_env = 17640, 24, 256
hidden_size, deepness = 512, 2
latent_dim = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sr = 44100
seed = seed_maker(frame_size, sr, N_filter_bank)
seed = seed.to(device)

#Dataset loading
dataset = VAE_Dataset(audio_path, frame_size, sr)
dataset_list = dataset.compute_dataset()
dataloader = DataLoader(dataset_list, batch_size=64, shuffle=True)

#Model loading
modelo = VAE_SubEnv(frame_size, hidden_size, deepness, latent_dim, N_filter_bank, param_per_env, seed, device)

#Optimizer
optimizer = optim.Adam(modelo.parameters(), lr=1e-3)

#Training
model_path = 'models/multiscale/water/'
train_multiscale_loss(modelo, optimizer, 100, device, dataloader, model_path)
model_path = 'models/statistics/water/'
train_statistics_loss(modelo, optimizer, 100, device, dataloader, model_path)

# Model 4 -------------------------------------------------------------------------------------

audio_path = 'sounds/water_augmented.wav'
model_path = 'models/water_deep/'

frame_size, N_filter_bank, param_per_env = 17640, 24, 256
hidden_size, deepness = 2048, 2
latent_dim = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sr = 44100
seed = seed_maker(frame_size, sr, N_filter_bank)
seed = seed.to(device)

#Dataset loading
dataset = VAE_Dataset(audio_path, frame_size, sr)
dataset_list = dataset.compute_dataset()
dataloader = DataLoader(dataset_list, batch_size=64, shuffle=True)

#Model loading
modelo = VAE_SubEnv(frame_size, hidden_size, deepness, latent_dim, N_filter_bank, param_per_env, seed, device)

#Optimizer
optimizer = optim.Adam(modelo.parameters(), lr=1e-3)

#Training
model_path = 'models/multiscale/water_deep/'
train_multiscale_loss(modelo, optimizer, 100, device, dataloader, model_path)
model_path = 'models/statistics/water_deep/'
train_statistics_loss(modelo, optimizer, 100, device, dataloader, model_path)
