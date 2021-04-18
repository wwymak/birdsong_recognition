import logging
import os
import warnings
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Audio, display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import kapre
import leaf_audio.frontend as frontend
import librosa
import soundfile as sf
import tensorflow as tf
import tensorflow_io as tfio
from kapre import STFT, Magnitude, MagnitudeToDecibel
from kapre.composed import get_log_frequency_spectrogram_layer, get_melspectrogram_layer
from librosa.display import specshow
from tensorflow import keras
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    GlobalAveragePooling2D,
    ReLU,
    Softmax,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import Sequence
from tqdm import tqdm
warnings.filterwarnings("ignore")

from leaf_audio.models import AudioClassifier, ConvNet, PANN, WaveGram
from data import path_to_audio, paths_and_labels_to_dataset

SHUFFLE_SEED = 101
BATCH_SIZE = 32


if __name__ == '__main__':
    assert len(tf.config.list_physical_devices('GPU')) >= 1
    dataset_folder = Path("/media/wwymak/Storage2/birdsong_dataset/xeno_canto_eu_cleaned")
    birdsong_metadata = pd.read_csv("xeno_canto_uk/xeno_canto_eu_cleaned.csv")
    birdsong_metadata["label_string"] = birdsong_metadata.en.apply(
        lambda x: x.lower().replace(" ", "_")
    )
    birdname_encoder = LabelEncoder()

    birdsong_metadata["label"] = birdname_encoder.fit_transform(
        birdsong_metadata.label_string
    )
    birdsong_metadata["filepath"] = birdsong_metadata[["id", "label_string"]].apply(
        lambda row: f"{dataset_folder}/{row.label_string}/{row['id']}.wav", axis=1
    )
    birdsong_metadata['valid'] = birdsong_metadata['filepath'].apply(lambda x: Path(x).exists())
    birdsong_metadata = birdsong_metadata[birdsong_metadata.valid == True]



    dataset = birdsong_metadata[birdsong_metadata.label_string.isin(['great_tit', 'eurasian_blackcap'])]
    dataset["label"] = dataset.label_string.apply(lambda x: 1 if x == 'great_tit' else 0)
    train_audio_paths, valid_audio_paths, train_labels, valid_labels = train_test_split(
        dataset[["filepath"]], dataset["label"], stratify=dataset["label"], test_size=0.2
    )

    train_ds = paths_and_labels_to_dataset(
        train_audio_paths.values.squeeze(), train_labels.values.squeeze()
    )
    train_ds = train_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(
        BATCH_SIZE
    )

    valid_ds = paths_and_labels_to_dataset(
        valid_audio_paths.values.squeeze(), valid_labels.values.squeeze()
    )
    valid_ds = valid_ds.shuffle(buffer_size=32 * 8, seed=SHUFFLE_SEED).batch(BATCH_SIZE)

    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    valid_ds = valid_ds.prefetch(tf.data.experimental.AUTOTUNE)

    convnet_encoder = ConvNet(
        filters=[64, 128, 256, 256, 512, 512],
        activation='relu',
        dropout=0.1
    )
    leaf_frontend = frontend.Leaf()

    model = AudioClassifier(frontend=leaf_frontend, encoder=convnet_encoder, num_outputs=2)
    learning_rate = 1e-4
    # input_shape = (SAMPLE_RATE * DURATION, NUM_CHANNELS)
    # model = build_model(input_shape, 2)

    # model.summary()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

    loss_fn = tf.keras.losses.BinaryCrossentropy()
    metric = 'accuracy'
    model.compile(loss=loss_fn,
                  optimizer=tf.keras.optimizers.Adam(learning_rate),
                  metrics=[metric])
    model.fit(train_ds,
              epochs=2,
              validation_data=valid_ds,
              callbacks=[tensorboard_callback])
