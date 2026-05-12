import torch
import torchaudio
import argparse
from pathlib import Path
from pystoi import stoi
from model.encoder_decoder import Encoder, Decoder
from model.rvq import ResidualVectorQuantizer

from inference import load_model_from_checkpoint, load_audio_from_path, run_trough_codec
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

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder, rvq = load_model_from_checkpoint(args.checkpoint, device)
    files = list(Path(args.test_dir).rglob("*.flac"))[:args.n_files]
    print(f"оцениваем на {len(files)} файлах")

    stoi_scores = []
    nisqa_scores = []

    for path in files:
        waveform = load_audio_from_path(path, device)
        reconstructed = run_trough_codec(waveform, encoder, decoder, rvq)

        stoi_score = compute_stoi(waveform, reconstructed)
        stoi_scores.append(stoi_score)

        if args.compute_nisqa:
            nisqa_score = compute_nisqa(reconstructed.cpu())
            nisqa_scores.append(nisqa_score)
            print(f"{path.name} | STOI: {stoi_score:.4f} | NISQA: {nisqa_score:.4f}")
        else:
            print(f"{path.name} | STOI: {stoi_score:.4f}")
        
    print(f"\nСредний STOI: {sum(stoi_scores)/len(stoi_scores):.4f}")
    if nisqa_scores:
        print(f"Средний NISQA: {sum(nisqa_scores)/len(nisqa_scores):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    type=str, required=True)
    parser.add_argument("--test_dir",      type=str, required=True)
    parser.add_argument("--n_files",       type=int, default=50)
    parser.add_argument("--compute_nisqa", action="store_true")
    main(parser.parse_args())
