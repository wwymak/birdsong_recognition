import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Audio
from scipy.io import wavfile
import tensorflow_io as tfio
import soundfile as sf

from sklearn.model_selection import train_test_split

from data import fetch_metadata,path_to_audio, paths_and_labels_to_dataset


class Yamnet:
    def __init__(self):
        self.model_path = 'https://tfhub.dev/google/yamnet/1'
        self.model = hub.load(self.model_path)
        class_map_path = self.model.class_map_path().numpy()
        self.yamnet_class_names = self.class_names_from_csv(class_map_path)

    def class_names_from_csv(self, class_map_csv_text):
        """Returns list of class names corresponding to score vector."""
        class_names = []
        with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
            reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])
        return class_names


    def extract_embedding(self, wav_data, label, fold):
        """ run YAMNet to extract embedding from the wav data """

        scores, embeddings, spectrogram = self.model(wav_data)
        num_embeddings = tf.shape(embeddings)[0]
        return (embeddings,
                tf.repeat(label, num_embeddings),
                tf.repeat(fold, num_embeddings))