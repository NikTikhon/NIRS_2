import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchaudio
from torchaudio import transforms
import os
from pathlib import Path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import glob
import torch.nn.functional as F
import torch.fft

torchaudio.backend.set_audio_backend(backend='soundfile')

class CustomDataset(Dataset):
    def __init__(self, dataset_path):
        self.imgs_path = dataset_path
        file_list = glob.glob(self.imgs_path + "*")
        # print(file_list)
        self.data = []
        for class_path in file_list:
            class_name = class_path.split("\\")[-1]
            for img_path in glob.glob(class_path + "/*.wav"):
                self.data.append([img_path, class_name])
        # print(self.data)
        self.class_map = os.listdir(self.imgs_path)
        self.img_dim = (416, 416)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        audio, rate = torchaudio.load(img_path)
        audio_fft = torch.fft.fft(audio, 16000)[0].reshape((1, 16000))
        # print(audio_fft.shape)
        # g = audio_fft.numpy()
        audio_fft = torch.abs(audio_fft)
        audio_fft = audio_fft[:, :8000]
        class_id = self.class_map.index(class_name)
        class_id = torch.tensor(class_id)
        return audio_fft, class_id


def main():
    dataset = CustomDataset()
    # audio =  torchaudio.load('E:\\16000_pcm_speeches_2\\audio\\p226\\out000.wav')
    data = DataLoader(dataset, shuffle=True, batch_size=32)
    for batch, (X, y) in enumerate(data):
        pass


if __name__ == '__main__':
    main()