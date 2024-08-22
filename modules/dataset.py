import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa

# Dataset maker -------------------------------------------------------------
class VAE_Dataset(Dataset):
    def __init__(self, audio_path, frame_size, sampling_rate):
        self.audio, _ = librosa.load(audio_path, sr=sampling_rate)
        onset_env = librosa.onset.onset_strength(y=self.audio, sr=sampling_rate)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sampling_rate, backtrack=False, pre_max=20, post_max=20, pre_avg=100, post_avg=100, delta=0.5, wait=0)
        self.onset_times = librosa.frames_to_time(onset_frames, sr=sampling_rate)
        self.frame_size = frame_size
        self.sampling_rate = sampling_rate

    def compute_dataset(self):
        actual_dataset = []
        time = self.frame_size
        y = self.audio
        for onset_time in self.onset_times:
            onset_frame = int(onset_time * self.sampling_rate)
            if onset_frame + time < len(y):
                segment=y[onset_frame:onset_frame + time]
                segment_tensor = torch.tensor(segment)
                actual_dataset.append(segment_tensor)
        return actual_dataset