import pandas as pd
import librosa
import soundfile
import os
import json
from pathlib import Path
from tqdm import tqdm

if __name__ == '__main__':
    birdsong_metadata = pd.read_csv('xeno_canto_eu_cleaned.csv')
    birdsong_metadata['label_string'] = birdsong_metadata.en.apply(lambda x: x.lower().replace(' ', '_'))
    input_dir = Path('/media/wwymak/Storage2/birdsong_dataset/xeno_canto_eu')
    output_dir = Path('/media/wwymak/Storage2/birdsong_dataset/xeno_canto_eu_cleaned')
    output_dir.mkdir(exist_ok=True)
    for category in birdsong_metadata.label_string.unique():
        (output_dir/category).mkdir(exist_ok=True)

    errors = []

    for idstr, label in tqdm(list(zip(birdsong_metadata['id'].values, birdsong_metadata.label_string.values))):
        input_file = input_dir / f"{idstr}.mp3"
        out_file = output_dir/label/f"{idstr}.wav"
        if out_file.exists():
            continue
        if not input_file.exists():
            continue
        audio, sr = librosa.load(input_file, sr=None)
        try:
            soundfile.write(out_file, audio, samplerate=sr)
        except Exception as e:
            errors.append({'file': str(input_file), 'error': e})

    with open('conversion_errors.json', 'w') as f:
        json.dump(errors, f)
