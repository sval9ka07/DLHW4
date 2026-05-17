import os
import io
import requests
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from IPython.display import Audio, display
from huggingface_hub import hf_hub_download

from inference import load_model_from_checkpoint, run_trough_codec


SAMPLE_RATE = 16000
N_FILES = 7
# чтобы не скачивать огромный датасет в Google Colab
# наверное, разрешено было делать в Kaggle
# но мне показалось, что обязатель все ipynb в Colab
TEST_CLEAN_URL = "https://huggingface.co/datasets/openslr/librispeech_asr/resolve/main/all/test.clean/0000.parquet"
DISFLUENCY_URL = "https://huggingface.co/datasets/nyrahealth/disfluency_speech_english/resolve/main/data/test-00000-of-00001.parquet"
# для анализа в test-clean я подумала, что будет неплохо взять более чисто звучащих дикторов
# иначе нет какой-то явной репрезентативности с последующим сраванением с зашумленными датасетами
# проверялось просто руками, то есть выводила с интервалом, в начале достаточно шумный диктор
GOOD_INDICES = [200, 233, 266, 300, 333, 366, 400]


# обертка над load_model_from_checkpoint для notebook
def load_model():
    checkpoint_path = hf_hub_download(
        repo_id="Finikita255/SoundStream-checkpoints",
        filename="checkpoints/checkpoint_final.pth",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder, rvq = load_model_from_checkpoint(checkpoint_path, device)
    return encoder, decoder, rvq, device

# в parquet хранится не только аудио, но и, например, связанный с ним текст
# поэтому нам придется, помимо скачивания, открыть файл, сделать из него словарь,
# найти там нужную колонку с аудио и читать его, расшифровывать bytes в числа
# для последующей обработки нейросетью
# здесь приходится самим, так как в dataset.py за нас это делает распаковщик,
# но для этого нужно скачать весь датасет или его подпапку, а это очень много
# а тут нам только кусочек датасета нужен

def load_from_parquet(url, parquet_path, indices, text_column, audio_column="audio"):
    if not os.path.exists(parquet_path):
        print(f"скачиваем часть датасета {parquet_path}")
        response = requests.get(url)
        with open(parquet_path, "wb") as f:
            f.write(response.content)
        print("скачали")
    else:
        print("parquet уже скачан, используем кэш")

    table = pq.read_table(parquet_path)
    data = table.to_pydict()

    samples = []
    for i in indices:
        audio_bytes = data["audio"][i]["bytes"]
        waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))

        # опять обработка на всякий случай
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        samples.append({
            "waveform": waveform,
            "text": data[text_column][i],
        })

        print(f"индекс {i}: {data[text_column][i][:60]} и т.д.")

    print(f"загружено {len(samples)} примеров")
    return samples

# обертки
def load_test_clean():
    return load_from_parquet(
        url=TEST_CLEAN_URL,
        parquet_path="asr_test/test_clean.parquet",
        indices=GOOD_INDICES,
        text_column="text",
    )

def load_external_english():
    return load_from_parquet(
        url=DISFLUENCY_URL,
        parquet_path="disfluency.parquet",
        indices=list(range(N_FILES)),
        text_column="verbatim_transcript",
    )

def plot_comparison(original, reconstructed, title):
    orig_np = original[0, 0].cpu().numpy()
    recon_np = reconstructed[0, 0].cpu().numpy()
    min_len = min(len(orig_np), len(recon_np))
    orig_np = orig_np[:min_len]
    recon_np = recon_np[:min_len]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(title, fontsize=13)

    t = np.arange(len(orig_np)) / SAMPLE_RATE
    axes[0, 0].plot(t, orig_np, linewidth=0.5)
    axes[0, 0].set_title("Waveform — оригинал")
    axes[0, 0].set_xlabel("время (с)")
    axes[0, 0].set_ylabel("амплитуда")

    axes[0, 1].plot(t, recon_np, linewidth=0.5, color="orange")
    axes[0, 1].set_title("Waveform — после кодека")
    axes[0, 1].set_xlabel("время (с)")
    axes[0, 1].set_ylabel("амплитуда")

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=80
    )
    to_db = torchaudio.transforms.AmplitudeToDB()

    orig_mel = to_db(mel_transform(torch.tensor(orig_np).unsqueeze(0)))
    recon_mel = to_db(mel_transform(torch.tensor(recon_np).unsqueeze(0)))

    axes[1, 0].imshow(orig_mel[0].numpy(), aspect="auto", origin="lower")
    axes[1, 0].set_title("Mel спектрограмма — оригинал")
    axes[1, 0].set_xlabel("фреймы")
    axes[1, 0].set_ylabel("mel bins")

    axes[1, 1].imshow(recon_mel[0].numpy(), aspect="auto", origin="lower")
    axes[1, 1].set_title("Mel спектрограмма — после кодека")
    axes[1, 1].set_xlabel("фреймы")
    axes[1, 1].set_ylabel("mel bins")

    plt.tight_layout()
    plt.show()
    return fig

def get_comparison_images(samples, encoder, decoder, rvq, device):
    results = []
    for i, sample in enumerate(samples):
        waveform = sample["waveform"].unsqueeze(0).to(device)
        reconstructed = run_trough_codec(waveform, encoder, decoder, rvq)

        fig = plot_comparison(
            waveform, reconstructed,
            title=f"Пример {i+1}: {sample['text'][:40]}..."
        )

        print("Оригинал:")
        display(Audio(waveform[0, 0].cpu().numpy(), rate=SAMPLE_RATE))
        print("После кодека:")
        display(Audio(reconstructed[0, 0].cpu().numpy(), rate=SAMPLE_RATE))

        results.append({
            "original": waveform[0, 0].cpu().numpy(),
            "reconstructed": reconstructed[0, 0].cpu().numpy(),
            "text": sample["text"],
            "figure": fig,
        })
    return results