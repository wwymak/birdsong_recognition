import tensorflow as tf
import numpy as np
import os
import pandas as pd
from pathlib import Path

import tensorflow_io as tfio
from tqdm import tqdm
import soundfile as sf

from yamnet import Yamnet

from scipy.signal import butter,filtfilt




class YamnetAudioClean:
    def __init__(self, threshold):
        """threshold sets the value of the bird audio class output from yamnet
         below which the audio section will be ignored"""
        self.threshold = threshold
        self.yamnet_base = Yamnet()
        yamnet_class_names = self.yamnet_base.yamnet_class_names
        self.bird_indices = [idx for idx in range(len(yamnet_class_names)) if yamnet_class_names[idx] in ['Bird',
             'Bird vocalization, bird call, bird song',
             'Chirp, tweet']]

    def extract_bird_sections(self, waveform_np, yamnet_scores_np):
        # each audio has sample rate 16000 hz, yamnet predicts on windows of 0.96 seconds long with each window 0.48 seconds apart.
        # we can assume each bin to be 0.48 seconds in our filtering
        bin_width = int(16000 * 0.48)
        valid_sects = np.unique(np.argwhere(yamnet_scores_np[:, self.bird_indices] >= 0.10)[:, 0])
        if len(valid_sects) == 0:
            return []
        mask = np.concatenate([np.arange(x * bin_width, (x + 1) * bin_width) for x in valid_sects])
        waveform_transformed = waveform_np[mask]
        return waveform_transformed

    # high pass filter from https://towardsdatascience.com/audio-onset-detection-data-preparation-for-a-baseball-application-using-librosa-7f9735430c17
    def butter_highpass(self, data, cutoff, fs, order=5):
        """
        Design a highpass filter.
        Args:
        - cutoff (float) : the cutoff frequency of the filter.
        - fs     (float) : the sampling rate.
        - order    (int) : order of the filter, by default defined to 5.
        """
        # calculate the Nyquist frequency
        nyq = 0.5 * fs
        # design filter
        high = cutoff / nyq
        b, a = butter(order, high, btype='high', analog=False)
        # returns the filter coefficients: numerator and denominator
        y = filtfilt(b, a, data)
        return y

    def load_wav_16k_mono(self,filepath):
        """ read in a waveform file and convert to 16 kHz mono """
        file_contents = tf.io.read_file(filepath)
        wav, sample_rate = tf.audio.decode_wav(
            file_contents,
            desired_channels=1)
        wav = tf.squeeze(wav, axis=-1)
        sample_rate = tf.cast(sample_rate, dtype=tf.int64)
        wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
        return wav

    def preprocess(self, filepath):
        waveform = self.load_wav_16k_mono(filepath)
        waveform = waveform.numpy()
        waveform = self.butter_highpass(waveform, 2000, 16000, order=5)
        scores, embeddings, spectrogram = self.yamnet_base.model(waveform)
        waveform = self.extract_bird_sections(waveform, scores.numpy())
        return waveform


if __name__ == '__main__':
    storage_path = Path('/media/wwymak/Storage2/birdsong_dataset/xeno_canto_eu_cleaned')
    output_folder = Path('/media/wwymak/Storage2/birdsong_dataset/xeno_canto_eu_cleaned_stage2')

    xeno_canto_dataset = pd.read_csv('xeno-canto_ukbirds_worldwide_songs_ratingAB.csv')
    xeno_canto_dataset["label_string"] = xeno_canto_dataset.en.apply(
        lambda x: x.lower().replace(" ", "_")
    )
    xeno_canto_dataset = xeno_canto_dataset[xeno_canto_dataset["label_string"].isin(os.listdir(storage_path))]
    xeno_canto_dataset["filepath"] = xeno_canto_dataset.apply(lambda row: str(storage_path / row.label_string / f"{row['id']}.wav"), axis=1)

    yamnet_cleaner = YamnetAudioClean(threshold=0.1)

    for label in xeno_canto_dataset.label_string.unique():
        (output_folder / label).mkdir(exist_ok=True)

    xeno_canto_dataset = xeno_canto_dataset.sample(frac=1.)
    errors = []
    for idx, row in tqdm(xeno_canto_dataset.iterrows(), total=len(xeno_canto_dataset)):
        try:
            waveform_processed = yamnet_cleaner.preprocess(row['filepath'])
            if len(waveform_processed) == 0:
                print(f"errro: {row['filepath']}")
                continue
            output_path = output_folder / row['label_string'] / f"{row['id']}.wav"
            sf.write(output_path, waveform_processed, samplerate=16000)
        except Exception as e:
            print(e)
            continue
