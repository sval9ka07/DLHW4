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
PARQUET_URL = "https://huggingface.co/datasets/openslr/librispeech_asr/resolve/main/all/test.clean/0000.parquet"

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

def load_test_clean():
    os.makedirs("asr_test", exist_ok=True)
    parquet_path = "asr_test/test_clean.parquet"

    if not os.path.exists(parquet_path):
        print("скачиваем часть датасета test_clean")
        response = requests.get(PARQUET_URL)
        with open(parquet_path, "wb") as f:
            f.write(response.content)
        print("скачали")
    else:
        print("parquet уже скачан, используем кэш")

    table = pq.read_table(parquet_path)
    data = table.to_pydict()

    samples = []
    for i in GOOD_INDICES:
        audio_bytes = data["audio"][i]["bytes"]
        waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))

        # опять обработка на всякий случай
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        samples.append({
            "waveform": waveform,
            "text": data["text"][i],
        })

        print(f"индекс {i}: {data['text'][i][:60]} и т.д.")

    print(f"загружено {len(samples)} примеров из LibriSpeech test-clean")
    return samples