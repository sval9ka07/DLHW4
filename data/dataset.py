import torch
import torchaudio
from torch.utils.data import Dataset
import torch.nn.functional as F
from pathlib import Path
import math

# ====================================================================================================
# Цитируем условие задания:
# Also, the SoundStream training setup might not be clear from the paper. Pretend like there was the following paragraph in the paper:
# For the training, we follow the SEANet optimization setup with a constant learning rate. 
# The SoundStream is trained for 45000 steps on 0.5s random crops of 16kHz audio with batch size 12 on Kaggle T4 GPU. 
# If the audio is shorter than 0.5s, we pad it with replication
# Note using 1s or 2s crops also works but may require more time/GPU memory. You can experiment.
# ====================================================================================================

SAMPLE_RATE = 16000 # просто диктуется количеством Гц
CROP = 0.5 # в секундах, нам порекомендовали сделать так
CROP_SAMPLES = math.ceil(SAMPLE_RATE * CROP) # 8000, но мы написали, откуда мы это взяли

class LibriSpeechDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.files = list(self.root_dir.rglob("*.flac"))

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path_to_file = self.files[idx]
        waveform, sample_rate = torchaudio.load(path_to_file)
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
        # Мы собираемся работать с моно аудио
        # но на всякий случай, если аудио формата стерео - усредняем каналы, переделывая в моно аудио
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        # Если аудио короче 0.5 секунд, то дополняем его, продлевая с padding='replicate' по условию задания.
        if waveform.shape[1] < CROP_SAMPLES:
            padding = CROP_SAMPLES - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding), 'replicate')
        start = torch.randint(0, waveform.shape[1] - CROP_SAMPLES + 1, (1,)).item()
        return waveform[:, start:start + CROP_SAMPLES]
    