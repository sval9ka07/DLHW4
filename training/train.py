# заглушка, чтобы обучение не засорять
import warnings
warnings.filterwarnings("ignore")
# раньше чем торч
from comet_ml import Experiment

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import argparse

from data.dataset import LibriSpeechDataset
from model.encoder_decoder import Encoder, Decoder
from model.rvq import ResidualVectorQuantizer

from model.discriminator import MultiScaleDiscriminator, STFTDiscriminator

from model.losses import discriminator_loss, adversarial_generator_loss, feature_matching_loss
from model.losses import multi_scale_spectral_reconstruction_loss, generator_loss

# ============================================================================================
# Цитируем условие задания (цитировали эту часть и в других файлах):
# For the training, we follow the SEANet optimization setup with a constant learning rate. 
# The SoundStream is trained for 45000 steps on 0.5s random crops of 16kHz audio with batch size 12 on Kaggle T4 GPU. 
# If the audio is shorter than 0.5s, we pad it with replication.
# ============================================================================================
# Смотрим в SEANet и цитируем:
# We train with the Adam optimizer, with a batch size of 
# 16 and a constant learning rate of 0.0001 with \beta_1 = 0.5 and \beta_2 = 0.9. 
# ============================================================================================

TOTAL_STEPS = 45000
BATCH_SIZE  = 12
LR          = 1e-4
BETAS       = (0.5, 0.9)

# ============================================================================================
# Из TABLE I в SoundStream берем C_enc = C_dec = 32 как дефолтную конфигурацию
# Потому что чтобы получить то же качество с меньшим C_enc = C_dec придется раздуть количество 
# параметров и проиграть в скорости обработки аудио (если я умею читать)
# --------------------------------------------------------------------------------------------
# Из TABLE II в SoundStream берем N_q=8, N=1024, так как на нем лучшее качество при 6kbps
# --------------------------------------------------------------------------------------------
# strides (2,4,5,5) при 16kHz вместо (2,4,5,8) при 24kHz по условию задания, но кажется это просто
# способ сохранить такой же битрейт, то есть
# 2*4*5*5 = 200 семпл/фрейм
# 16000 / 200 = 80 фреймов/сек
# N_q=8 квантайзеров, каждый кодбук N=1024=2^10 (про логарифм 2 в части RVQ в статье)
# 8 * 10 = 80 бит/фрейм
# 80 * 80 = 6400 бит / сек - около 6kbps, чисто эмпирически оставляем такие параметры
# ============================================================================================

C   = 32
D   = 64 # возможно нужно будет менять, но пока так
N_Q = 8
N   = 1024

LOG_EVERY  = 100
SAVE_EVERY = 5000

# ============================================================================================
# В статье используем секцию E. Training objective
# Там написнао, в каком порядке идут шаги, где мы обучаем, а где просто делаем прогон 
# для генератора или дискриминатора, какие используем loss на каждом шаге.
# ============================================================================================

def infinite_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch

def log_step(experiment, logs, step):
    print(f"step {step} | " + " | ".join(f"{key}: {value:.4f}" for key, value in logs.items()))
    if experiment is not None:
        experiment.log_metrics(logs, step=step)

def log_audio_samples(experiment, encoder, decoder, rvq, x, step):
    if experiment is None:
        return
    encoder.eval()
    decoder.eval()
    rvq.eval()
    with torch.no_grad():
        sample_hat = decoder(rvq(encoder(x[:1]))[1])
    experiment.log_audio(
        x[0, 0].cpu().numpy(), sample_rate=16000,
        file_name=f"original_step{step}.wav"
    )
    experiment.log_audio(
        sample_hat[0, 0].cpu().numpy(), sample_rate=16000,
        file_name=f"reconstructed_step{step}.wav"
    )
 
def main(args):
    # Comet
    experiment = None
    if args.use_comet:
        experiment = Experiment(
            api_key=args.comet_apikey,
            project_name=args.comet_project,
            auto_metric_logging=False,
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    dataset = LibriSpeechDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    data_iter = infinite_dataloader(dataloader)
    print(f"В датасете {len(dataset)} файлов")

    encoder = Encoder(C, D).to(device)
    decoder = Decoder(C, D).to(device)
    rvq     = ResidualVectorQuantizer(N_Q, N, D).to(device)

    wave_disc = MultiScaleDiscriminator().to(device)
    stft_disc = STFTDiscriminator(C=32).to(device)

    optimizer_generator = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + list(rvq.parameters()),
        lr=LR, betas=BETAS
    )

    optimizer_discriminator = torch.optim.Adam(
        list(wave_disc.parameters()) + list(stft_disc.parameters()),
        lr=LR, betas=BETAS
    )

    for step in range(TOTAL_STEPS):
        batch = next(data_iter)
        x = batch.to(device) # тензор (B, 1, 8000)

        # Шаг генератора
        encoder.train()
        decoder.train()
        rvq.train()
        wave_disc.eval()
        stft_disc.eval()

        emb = encoder(x)
        _, discrete_emb, commitment_loss = rvq(emb)
        x_hat = decoder(discrete_emb)

        with torch.no_grad():
            original_feats_g = wave_disc(x) + [stft_disc(x)]
        reconstructed_feats_g = wave_disc(x_hat) + [stft_disc(x_hat)]

        g_total_loss, logs = generator_loss(x, x_hat, original_feats_g, reconstructed_feats_g, commitment_loss)
        optimizer_generator.zero_grad()
        g_total_loss.backward()
        optimizer_generator.step()

        # Шаг дискриминатора
        wave_disc.train()
        stft_disc.train()
        original_feats_d = wave_disc(x) + [stft_disc(x)]
        reconstructed_feats_d = wave_disc(x_hat.detach()) + [stft_disc(x_hat.detach())]
        d_total_loss = discriminator_loss(original_feats_d, reconstructed_feats_d).mean()

        optimizer_discriminator.zero_grad()
        d_total_loss.backward()
        optimizer_discriminator.step()

        # Логирование
        if step % LOG_EVERY == 0:
            logs['loss_d'] = d_total_loss.item()
            log_step(experiment, logs, step)
        if step % (LOG_EVERY * 10) == 0:
            log_audio_samples(experiment, encoder, decoder, rvq, x, step)

# что должно быть у args
# data_dir -  путь к данным
# use_comet - используем comet или нет
# comet_apikey - апиключ комет
# comet_project - название проекта комет
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--use_comet",     action="store_true")
    parser.add_argument("--comet_apikey", type=str, default=None)
    parser.add_argument("--comet_project", type=str, default="soundstream")
    main(parser.parse_args())