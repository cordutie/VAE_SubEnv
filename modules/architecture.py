# Architecture
import torch
import torch.nn as nn
from  modules.synthesizer import TextEnv, TextEnv_batches

def mlp(in_size, hidden_size, n_layers):
    channels = [in_size] + [hidden_size] * n_layers
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(channels[i], channels[i + 1]))
        net.append(nn.LayerNorm(channels[i + 1]))
        net.append(nn.LeakyReLU())
    return nn.Sequential(*net)

def gru(n_input, hidden_size):
    return nn.GRU(n_input, hidden_size, batch_first=True)

class VAE_SubEnv(nn.Module):
    def __init__(self, frame_size, hidden_size, deepness, latent_dim, N_filter_bank, param_per_env, seed, device):
        super(VAE_SubEnv, self).__init__()
        self.seed = seed
        self.device = device
        
        # parameters
        self.N_filter_bank = N_filter_bank
        self.param_per_env = param_per_env
        
        # encoder
        self.encoder_1 = mlp(frame_size, hidden_size, deepness)
        self.encoder_mean = nn.Linear(hidden_size, latent_dim)
        self.encoder_logvar  = nn.Linear(hidden_size, latent_dim)
        
        # decoder    
        self.a_decoder_1 = mlp(latent_dim, hidden_size, deepness)
        self.a_decoder_2 = nn.Linear(hidden_size, N_filter_bank * param_per_env)
        self.p_decoder_1 = mlp(latent_dim, hidden_size, deepness)
        self.p_decoder_2 = nn.Linear(hidden_size, N_filter_bank * param_per_env)

    def encode(self, x):
        x = self.encoder_1(x)
        mean, logvar = self.encoder_mean(x), self.encoder_logvar(x)
        var = torch.exp(logvar)
        return mean, var

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)      
        z = mean + torch.sqrt(var)*epsilon
        return z

    def decode(self, z):
        a = self.a_decoder_1(z)
        a = self.a_decoder_2(a)
        a = torch.sigmoid(a)
        p = self.p_decoder_1(z)
        p = self.p_decoder_2(p)
        p = 2 * torch.pi * torch.sigmoid(p)
        real_param = a * torch.cos(p)
        imag_param = a * torch.sin(p)
        return real_param, imag_param

    def forward(self, x):
        # print("x:    ",x.shape)
        mean, var = self.encode(x)
        # print("mean: ",mean.shape)
        # print("var:  ",var.shape)
        z = self.reparameterization(mean, var)
        # print("z:    ",z.shape)
        real_param, imag_param = self.decode(z)
        # print(real_param.shape)
        # print(imag_param.shape)
        x_hat = TextEnv_batches(real_param, imag_param, self.seed)
        # print(x_hat.shape)
        return x_hat, mean, var
    
    def generate(self, z):
        # print("z:    ",z.shape)
        real_param, imag_param = self.decode(z)
        # print(real_param.shape)
        # print(imag_param.shape)
        x_hat = TextEnv(real_param, imag_param, self.seed)
        return x_hat

# Function to save the model
def save_model(model, path):
    torch.save(model.state_dict(), path)
    # print(f'Model saved to {path}')

# Function to load the model
def load_model(path,   frame_size, hidden_size, deepness, latent_dim, N_filter_bank, param_per_env, seed, device):
    model = VAE_SubEnv(frame_size, hidden_size, deepness, latent_dim, N_filter_bank, param_per_env, seed, device).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    print(f'Model loaded from {path}')
    return model