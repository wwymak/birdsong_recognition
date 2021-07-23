from ast import literal_eval
import pandas as pd
import requests
import librosa
import soundfile
from pathlib import Path
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':

    xeno_canto_dataset = pd.read_csv('xeno-canto_ukbirds_worldwide_songs_ratingAB.csv')
    xeno_canto_dataset["label_string"] = xeno_canto_dataset.en.apply(
        lambda x: x.lower().replace(" ", "_")
    )
    xeno_canto_dataset['download_url'] = xeno_canto_dataset.file.apply(lambda x: f"https:{x}")

    storage_path = Path('/media/wwymak/Storage2/birdsong_dataset/xeno_canto_eu_cleaned')
    errors = []
    for id, label_string, url in tqdm(zip(xeno_canto_dataset['id'], xeno_canto_dataset['label_string'], xeno_canto_dataset['download_url']), total=len(xeno_canto_dataset)):
        if (storage_path / label_string / f"{id}.wav").exists():
            continue
        else:
            r = requests.get(url)
            if r.status_code == 200:
                with open('temp.mp3', 'wb') as f:
                    f.write(r.content)

                audio, sr = librosa.load('temp.mp3', sr=None)
                try:
                    out_file = storage_path / label_string / f"{id}.wav"
                    soundfile.write(out_file, audio, samplerate=sr)
                except Exception as e:
                    errors.append({'file': id, 'error': e})
            else:
                errors.append(id)
    pd.Series(errors).to_csv('xeno_canto_eu_errors.csv')