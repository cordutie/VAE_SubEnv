import torch

# MULTISCALE SPECTOGRAM HERE
def multiscale_fft(signal, scales=[4096, 2048, 1024, 512, 256, 128], overlap=.75):
    stfts = []
    for s in scales:
        S = torch.stft(
            signal,
            s,
            int(s * (1 - overlap)),
            s,
            torch.hann_window(s).to(signal),
            True,
            normalized=True,
            return_complex=True,
        ).abs()
        stfts.append(S)
    return stfts

def multiscale_spectrogram_loss(x, x_hat):
    ori_stft = multiscale_fft(x)
    rec_stft = multiscale_fft(x_hat)
    loss = 0
    for s_x, s_y in zip(ori_stft, rec_stft):
        lin_loss = (s_x - s_y).abs().mean()
        log_loss = (safe_log(s_x) - safe_log(s_y)).abs().mean()
        loss = loss + lin_loss + log_loss
    return loss

def safe_log(x):
    return torch.log(x + 1e-7)

def loss_function(x, x_hat, mean, var):
    reproduction_loss = multiscale_spectrogram_loss(x, x_hat)
    KLD = - 0.5 * torch.sum(1+ safe_log(var) - mean.pow(2) - var)
    # print("Reproduction Loss: ", reproduction_loss)
    # print("KLD: ", KLD)
    return reproduction_loss + KLD