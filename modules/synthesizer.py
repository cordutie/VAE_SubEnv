# Synthesizers
import torch

def TextEnv(parameters_real, parameters_imag, seed, target_loudness=1):
    size          = seed.shape[0]
    N_filter_bank = seed.shape[1]
    
    N = parameters_real.size(0)
    parameters_size = N // N_filter_bank
    signal_final = torch.zeros(size, dtype=torch.float32)
    
    for i in range(N_filter_bank):
        # Construct the local parameters as a complex array
        parameters_local = parameters_real[i * parameters_size : (i + 1) * parameters_size] + 1j * parameters_imag[i * parameters_size : (i + 1) * parameters_size]
        
        # Initialize FFT coefficients array
        fftcoeff_local = torch.zeros(int(size/2)+1, dtype=torch.complex64)
        fftcoeff_local[:parameters_size] = parameters_local ###########################################3
        
        # Compute the inverse FFT to get the local envelope
        env_local = torch.fft.irfft(fftcoeff_local)
        
        # Extract the local noise
        noise_local = seed[:, i]
        
        # Generate the texture sound by multiplying the envelope and noise
        texture_sound_local = env_local * noise_local
        
        # Accumulate the result
        signal_final += texture_sound_local
    
    loudness = torch.sqrt(torch.mean(signal_final ** 2))
    signal_final = signal_final / loudness

    signal_final = target_loudness * signal_final

    return signal_final

def TextEnv_batches(parameters_real, parameters_imag, seed):
    size          = seed.shape[0]
    N_filter_bank = seed.shape[1]
    
    # Get the batch size
    batch_size = parameters_real.size(0)
    parameters_size = parameters_real.size(1) // N_filter_bank

    # Initialize the final signal tensor for the entire batch
    signal_final = torch.zeros((batch_size, size), dtype=torch.float32, device=parameters_real.device)
    
    for i in range(N_filter_bank):
        # Construct the local parameters as a complex array for each filter in the batch
        parameters_local = (parameters_real[:, i * parameters_size : (i + 1) * parameters_size] 
                            + 1j * parameters_imag[:, i * parameters_size : (i + 1) * parameters_size])
        
        # Initialize FFT coefficients array for the entire batch
        fftcoeff_local = torch.zeros((batch_size, int(size / 2) + 1), dtype=torch.complex64, device=parameters_real.device)
        fftcoeff_local[:, :parameters_size] = parameters_local

        # Compute the inverse FFT to get the local envelope for each batch item
        env_local = torch.fft.irfft(fftcoeff_local).real

        # Extract the local noise for each batch item
        noise_local = seed[:, i]

        # Generate the texture sound by multiplying the envelope and noise for each batch item
        texture_sound_local = env_local * noise_local

        # Accumulate the result for each batch item
        signal_final += texture_sound_local
    
    return signal_final