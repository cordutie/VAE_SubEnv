import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import modules.filterbanks as fb
# from IPython.display import Audio, display

# Dataset maker -------------------------------------------------------------
class VAE_Dataset(Dataset):
    def __init__(self, audio_path, frame_size, sampling_rate, N_filter_bank):
        self.audio, _ = librosa.load(audio_path, sr=sampling_rate)
        self.audio = librosa.effects.preemphasis(self.audio)
        onset_env = librosa.onset.onset_strength(y=self.audio, sr=sampling_rate)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sampling_rate, backtrack=False, pre_max=20, post_max=20, pre_avg=100, post_avg=100, delta=0.3, wait=0)
        self.onset_times = librosa.frames_to_time(onset_frames, sr=sampling_rate)
        self.frame_size = frame_size
        self.sampling_rate = sampling_rate
        self.filterbank = fb.EqualRectangularBandwidth(frame_size, sampling_rate, N_filter_bank, 20, sampling_rate // 2)

    def compute_dataset(self):
        actual_dataset = []
        time = self.frame_size
        # window = np.power(np.hamming(self.frame_size), 0.01)
        y = self.audio
        for onset_time in self.onset_times:
            onset_frame = int(onset_time * self.sampling_rate)
            if onset_frame + time < len(y):
                segment=y[onset_frame:onset_frame + time]
                # # windowing
                # segment = segment * window
                # segment to tensor dtype float
                segment_tensor = torch.tensor(segment, dtype=torch.float32)
                #normalization
                segment_tensor = (segment_tensor - torch.mean(segment_tensor)) / torch.std(segment_tensor)
                sub_bands = self.filterbank.generate_subbands(segment_tensor)[:, 1:-1]
                # energy in each subband
                energy_bands = torch.sqrt(torch.sum(sub_bands**2, dim=0))
                actual_dataset.append([segment_tensor, energy_bands])
        # actual_dataset = actual_dataset[2:3]
        return actual_dataset
    
    # def play_dataset(self, actual_dataset, sr):
    #     actual_dataset = actual_dataset[16:20]
    #     # Play the dataset using Display() and audio()
    #     for i in range(len(actual_dataset)):
    #         display(Audio(actual_dataset[i][0].numpy(), rate=sr))
