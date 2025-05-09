import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio as T
import torchaudio.transforms as TT
import torch.nn.functional as F


class AudioDataset(Dataset):
    def __init__(self, file_dir, sr=22050, hop_samples=256, n_fft=1024, n_mels=80,
                 target_time_steps=2580, save_features=True, feature_save_dir=None):
        self.file_dir = file_dir
        self.files = [os.path.join(file_dir, f) for f in os.listdir(file_dir) if f.endswith(".wav")]

        self.sr = sr
        self.hop_samples = hop_samples
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.target_time_steps = target_time_steps
        self.save_features = save_features

        if feature_save_dir is not None:
            os.makedirs(feature_save_dir, exist_ok=True)
        self.feature_save_dir = feature_save_dir

        self.mel_transform = TT.MelSpectrogram(
            sample_rate=sr,
            win_length=hop_samples * 4,
            hop_length=hop_samples,
            n_fft=n_fft,
            f_min=20.0,
            f_max=sr / 2.0,
            n_mels=n_mels,
            power=1.0,
            normalized=True,
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        mel_features = self._process_file(file_path)
    
        if self.save_features and self.feature_save_dir:
            base_filename = os.path.splitext(os.path.basename(file_path))[0]
            save_path = os.path.join(self.feature_save_dir, f"{base_filename}_mel.npy")
            np.save(save_path, mel_features.numpy())
        else:
            base_filename = os.path.basename(file_path)  
    
        return mel_features, base_filename


    def _process_file(self, file_path):
        audio, sr = T.load(file_path)
        audio = torch.clamp(audio[0], -1.0, 1.0)

        if self.sr != sr:
            raise ValueError(f'Invalid sample rate {sr} for file {file_path}.')

        mel_spec = self._extract_mel(audio)
        mel_spec = self._adjust_time_steps(mel_spec)
        return mel_spec

    def _extract_mel(self, audio):
        with torch.no_grad():
            mel_spec = self.mel_transform(audio)
            mel_spec = 20 * torch.log10(torch.clamp(mel_spec, min=1e-5)) - 20
            mel_spec = torch.clamp((mel_spec + 100) / 100, 0.0, 1.0)
        return mel_spec

    def _adjust_time_steps(self, features):
        current_time_steps = features.shape[1]
        target_time_steps = self.target_time_steps

        if current_time_steps < target_time_steps:
            features = F.pad(features, (0, target_time_steps - current_time_steps))
        elif current_time_steps > target_time_steps:
            features = features[:, :target_time_steps]

        return features


if __name__ == '__main__':
    input_dir = "/home/quincy/DATA/music/dataset/rock"
    output_dir = "/home/quincy/DATA/music/features/rock"
    dataset = AudioDataset(input_dir, target_time_steps=2580,
                           save_features=True, feature_save_dir=output_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for idx, data in enumerate(dataloader):
        print(f"Sample {idx} - Mel Feature Shape: {data.shape}")
        # if idx == 1:
        #     break
