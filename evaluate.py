import torch
import torchaudio
import argparse
from pathlib import Path
from pystoi import stoi
from model.encoder_decoder import Encoder, Decoder
from model.rvq import ResidualVectorQuantizer

from inference import load_model_from_checkpoint
C, D, N_Q, N = 32, 64, 8, 1024
SAMPLE_RATE = 16000

# требуется и оригинал и выход кодека
def compute_stoi(original, reconstructed):
    orig_np = original[0, 0].cpu().numpy()
    recon_np = reconstructed[0, 0].cpu().numpy()
    # на всякий случай, если вдруг они разной длины
    min_len = min(len(orig_np), len(recon_np))
    return stoi(orig_np[:min_len], recon_np[:min_len], SAMPLE_RATE, extended=False)


# метрика, не требующая оригинала
def compute_nisqa(reconstructed):
    from torchmetrics.audio.nisqa import NonIntrusiveSpeechQualityAssessment
    nisqa = NonIntrusiveSpeechQualityAssessment(fs=SAMPLE_RATE)
    return nisqa(reconstructed[0]).item()

