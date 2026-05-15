import torch
import torchaudio
import argparse
from pathlib import Path

from model.encoder_decoder import Encoder, Decoder
from model.rvq import ResidualVectorQuantizer

C, D, N_Q, N = 32, 64, 8, 1024
SAMPLE_RATE = 16000

def load_model_from_checkpoint(checkpoint_path, device):
    encoder = Encoder(C, D).to(device)
    decoder = Decoder(C, D).to(device)
    rvq     = ResidualVectorQuantizer(N_Q, N, D).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    rvq.load_state_dict(checkpoint["rvq"])

    encoder.eval()
    decoder.eval()
    rvq.eval()
    return encoder, decoder, rvq

# ==============================================================================================
# Цитируем задание:
# 4. Train on audio crops, evaluate on full audio.
# ==============================================================================================
# Будем использовать эту функцию и в evaluate
def load_audio_from_path(path, device):
    waveform, sr = torchaudio.load(path)
    # на всякий случай, если герцовка другая
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    # на всякий случай, если аудио не моно
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform.unsqueeze(0).to(device)

def run_trough_codec(waveform, encoder, decoder, rvq):
    with torch.no_grad():
        z = encoder(waveform)
        _, z_q, _ = rvq(z)
        reconstructed = decoder(z_q)
    return reconstructed

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    encoder, decoder, rvq = load_model_from_checkpoint(args.checkpoint, device)
    input_files = list(Path(args.input_dir).rglob("*.flac")) + \
                  list(Path(args.input_dir).rglob("*.wav"))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for path in input_files:
        waveform = load_audio_from_path(path, device)
        reconstructed = run_trough_codec(waveform, encoder, decoder, rvq)
        out_path = Path(args.output_dir) / (path.stem + "_reconstructed.wav")
        torchaudio.save(str(out_path), reconstructed[0].cpu(), SAMPLE_RATE)
        print(f"{path.name} → {out_path.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_dir",  type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output")
    main(parser.parse_args())