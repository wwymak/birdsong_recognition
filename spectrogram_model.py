import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Audio
from pathlib import Path

import tensorflow_io as tfio
from tqdm import tqdm
import soundfile as sf

from sklearn.model_selection import train_test_split
from functools import partial

batch_size = 64
steps_per_epoch = 50
AUTOTUNE = tf.data.experimental.AUTOTUNE
num_classes = 84
def get_mfcc(waveform, sample_rate):
    spectrogram = tfio.experimental.audio.spectrogram(
        waveform, nfft=512, window=512, stride=256)
    mel_spectrogram = tfio.experimental.audio.melscale(
        spectrogram, rate=sample_rate, mels=128, fmin=2000, fmax=sample_rate//2)
    dbscale_mel_spectrogram = tfio.experimental.audio.dbscale(
        mel_spectrogram, top_db=80)
    return tf.expand_dims(dbscale_mel_spectrogram, axis=2)

def parse_tfrecord(example):
    feature_description = {
        "audio": tf.io.VarLenFeature(tf.float32),
        "category_id": tf.io.FixedLenFeature([], tf.int64),
    }

    example = tf.io.parse_single_example(example, feature_description)
    example['audio'] = tf.sparse.to_dense(example['audio'])
    return example

def prepare_sample(features):
    audio = get_mfcc(features['audio'], 16000)
    return audio, features['category_id']

def get_dataset(filenames, batch_size):
    dataset = (
        tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
        .map(parse_tfrecord, num_parallel_calls=AUTOTUNE)
        .map(prepare_sample, num_parallel_calls=AUTOTUNE)
        .shuffle(batch_size * 10)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    return dataset
# correct version of freq mask as per https://github.com/tensorflow/io/pull/1170
def freq_mask(input, param, name=None):
    """
    Apply masking to a spectrogram in the freq domain.
    Args:
      input: An audio spectogram.
      param: Parameter of freq masking.
      name: A name for the operation (optional).
    Returns:
      A tensor of spectrogram.
    """
    input = tf.convert_to_tensor(input)
    # TODO: Support audio with channel > 1.
    freq_max = tf.shape(input)[1]
    f = tf.random.uniform(shape=(), minval=0, maxval=param, dtype=tf.dtypes.int32)
    f0 = tf.random.uniform(
        shape=(), minval=0, maxval=freq_max - f, dtype=tf.dtypes.int32
    )
    indices = tf.reshape(tf.range(freq_max), (1, -1))
    condition = tf.math.logical_and(
        tf.math.greater_equal(indices, f0), tf.math.less(indices, f0 + f)
    )
    return tf.where(condition, tf.cast(0, input.dtype), input)


def prepare_sample_specaugment(features):
    audio = get_mfcc(features['audio'], 16000)
    augmented = freq_mask(audio, 10)
    return augmented, features['category_id']

def get_dataset_augment(filenames, batch_size):
    dataset = (
        tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
        .map(parse_tfrecord, num_parallel_calls=AUTOTUNE)
        .map(prepare_sample_specaugment, num_parallel_calls=AUTOTUNE)
        .shuffle(batch_size)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    return dataset

if __name__ == '__main__':
    assert len(tf.config.list_physical_devices('GPU')) >= 1
    tfrecords_dir = Path('/media/wwymak/Storage2/birdsong_dataset/xeno_canto_eu_cleaned_stage2_tfrecord')
    train_filenames = tf.io.gfile.glob(f"{str(tfrecords_dir)}/*_train.tfrec")
    valid_filenames = tf.io.gfile.glob(f"{str(tfrecords_dir)}/*_val.tfrec")
    test_filenames = tf.io.gfile.glob(f"{str(tfrecords_dir)}/*_test.tfrec")

    train_dataset_augment = get_dataset_augment(train_filenames, batch_size)
    valid_dataset_augment = get_dataset_augment(valid_filenames, batch_size)
    # test_dataset_augment = get_dataset_augment(test_filenames, batch_size)

    norm_layer = tf.keras.layers.experimental.preprocessing.Normalization()
    norm_layer.adapt(train_dataset_augment.map(lambda x, _: x))

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(625, 128, 1)),
        tf.keras.layers.experimental.preprocessing.Resizing(256, 128),
        norm_layer,
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(84),
    ])

    model.summary()

    metrics = [
        tf.keras.metrics.SparseTopKCategoricalAccuracy(
            k=5, name="sparse_top_5_categorical_accuracy", dtype=None
        ),
        'accuracy'
    ]

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath="models/spectrogram_allbirds",
            save_best_only=True,  # Only save a model if `val_loss` has improved.
            monitor="val_sparse_top_5_categorical_accuracy",
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir="./logs",
        ),  # How often to write logs (default: once per epoch)
        tf.keras.callbacks.EarlyStopping(
            monitor="val_sparse_top_5_categorical_accuracy",
            min_delta=0,
            patience=5,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        )
    ]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=metrics,
    )

    history = model.fit(
        train_dataset_augment,
        validation_data=valid_dataset_augment,
        epochs=50,
        callbacks=callbacks,
    )
