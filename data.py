import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import os

SAMPLING_RATE = 16_000

bird_mapping_idx_to_name = {0: 'barn_swallow', 1: 'black-headed_gull', 2: 'black_woodpecker', 3: 'carrion_crow', 4: 'coal_tit',
                5: 'common_blackbird', 6: 'common_chaffinch', 7: 'common_chiffchaff', 8: 'common_cuckoo',
                9: 'common_house_martin', 10: 'common_linnet', 11: 'common_moorhen', 12: 'common_nightingale',
                13: 'common_pheasant', 14: 'common_redpoll', 15: 'common_redshank', 16: 'common_redstart',
                17: 'common_reed_bunting', 18: 'common_snipe', 19: 'common_starling', 20: 'common_swift',
                21: 'common_whitethroat', 22: 'common_wood_pigeon', 23: 'corn_bunting', 24: 'dunlin', 25: 'dunnock',
                26: 'eurasian_blackcap', 27: 'eurasian_blue_tit', 28: 'eurasian_bullfinch',
                29: 'eurasian_collared_dove', 30: 'eurasian_coot', 31: 'eurasian_golden_oriole', 32: 'eurasian_jay',
                33: 'eurasian_magpie', 34: 'eurasian_nuthatch', 35: 'eurasian_oystercatcher',
                36: 'eurasian_reed_warbler', 37: 'eurasian_skylark', 38: 'eurasian_tree_sparrow',
                39: 'eurasian_treecreeper', 40: 'eurasian_wren', 41: 'eurasian_wryneck', 42: 'european_bee-eater',
                43: 'european_golden_plover', 44: 'european_goldfinch', 45: 'european_green_woodpecker',
                46: 'european_greenfinch', 47: 'european_herring_gull', 48: 'european_honey_buzzard',
                49: 'european_nightjar', 50: 'european_robin', 51: 'european_turtle_dove', 52: 'garden_warbler',
                53: 'goldcrest', 54: 'great_spotted_woodpecker', 55: 'great_tit', 56: 'grey_partridge',
                57: 'house_sparrow', 58: 'lesser_whitethroat', 59: 'long-tailed_tit', 60: 'marsh_tit',
                61: 'marsh_warbler', 62: 'meadow_pipit', 63: 'northern_lapwing', 64: 'northern_raven',
                65: 'red-throated_loon', 66: 'red_crossbill', 67: 'redwing', 68: 'river_warbler', 69: 'rock_dove',
                70: 'rook', 71: 'sedge_warbler', 72: 'song_thrush', 73: 'spotted_flycatcher', 74: 'stock_dove',
                75: 'tawny_owl', 76: 'tree_pipit', 77: 'western_yellow_wagtail', 78: 'willow_ptarmigan',
                79: 'willow_tit', 80: 'willow_warbler', 81: 'wood_sandpiper', 82: 'wood_warbler', 83: 'yellowhammer'}

bird_mapping_name_to_idx = {v:k for k, v in bird_mapping_idx_to_name.items()}


def paths_and_labels_to_dataset(audio_paths, labels):
    """Constructs a dataset of audios and labels."""
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    audio_ds = path_ds.map(lambda x: tf.py_function(path_to_audio, [x], [tf.float32]))
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((audio_ds, label_ds))


def path_to_audio(path, output_sample_rate=16_000, audio_length=30):
    """Reads and decodes an audio file."""
    audio = tf.io.read_file(path)
    audio, sample_rate = tf.audio.decode_wav(audio)
    # audio = tf.squeeze(audio, axis=-1)
    audio = tf.cast(audio, tf.float32)
    audio = tfio.audio.resample(
        audio, rate_in=tf.cast(sample_rate, tf.int64), rate_out=output_sample_rate, name=None
    )
    effective_length = output_sample_rate * audio_length

    if audio.shape[0] < effective_length:
        padding=(effective_length - audio.shape[0])
        padding_start =  padding// 2
        padding_end = padding- padding_start

        paddings = tf.constant([[padding_start, padding_end ], [0,0]])
        # 'constant_values' is 0.
        # rank of 't' is 2.
        audio = tf.pad(audio, paddings, "CONSTANT")
    elif audio.shape[0] > effective_length:
        start = np.random.randint(low=0, high=(audio.shape[0] - effective_length))

        audio = audio[start:start+effective_length, :]
    audio = tf.squeeze(audio, axis=[-1])
    return audio


def fetch_metadata():
    dataset_folder = Path("/media/wwymak/Storage2/birdsong_dataset/xeno_canto_eu_cleaned")
    birdsong_metadata = pd.read_csv("xeno_canto_uk/xeno_canto_eu_cleaned.csv")
    birdsong_metadata["label_string"] = birdsong_metadata.en.apply(
        lambda x: x.lower().replace(" ", "_")
    )
    birdsong_metadata = birdsong_metadata[birdsong_metadata.label_string.isin((birdsong_metadata.label_string.value_counts() > 100).index)]

    birdsong_metadata["label"] = birdsong_metadata.label_string.apply(lambda x:bird_mapping_name_to_idx.get(x)).astype(int)

    birdsong_metadata["filepath"] = birdsong_metadata[["id", "label_string"]].apply(
        lambda row: f"{dataset_folder}/{row.label_string}/{row['id']}.wav", axis=1
    )
    birdsong_metadata['valid'] = birdsong_metadata['filepath'].apply(lambda x: Path(x).exists())
    birdsong_metadata = birdsong_metadata[birdsong_metadata.valid == True]

    return birdsong_metadata

def fetch_metadata_v2():
    dataset_folder = Path('/media/wwymak/Storage2/birdsong_dataset/xeno_canto_eu_cleaned_stage2')

    birdsong_metadata = pd.read_csv('xeno_canto_uk/xeno-canto_ukbirds_worldwide_songs_ratingAB.csv')
    birdsong_metadata["label_string"] = birdsong_metadata.en.apply(
        lambda x: x.lower().replace(" ", "_")
    )

    birdsong_metadata = birdsong_metadata[birdsong_metadata["label_string"].isin(os.listdir(dataset_folder))]
    birdsong_metadata["filepath"] = birdsong_metadata.apply(lambda row: str(dataset_folder / row.label_string / f"{row['id']}.wav"), axis=1)
    birdsong_metadata['is_valid'] = birdsong_metadata["filepath"].apply(lambda x: Path(x).exists())
    label_counts = birdsong_metadata.label_string.value_counts()
    valid_labels = label_counts[label_counts > 10].index
    birdsong_metadata = birdsong_metadata[birdsong_metadata.label_string.isin(valid_labels)]
    birdsong_metadata = birdsong_metadata[birdsong_metadata['is_valid']]

    le = LabelEncoder()
    birdsong_metadata['label'] = le.fit_transform(birdsong_metadata['label_string'])

    bird_mapping_idx_to_name = {k: v for k, v in zip(np.arange(len(le.classes_)), le.classes_)}
    return birdsong_metadata, bird_mapping_idx_to_name

def db_to_linear(samples):
    return 10.0 ** (samples / 20.0)


def loudness_normalization(samples: tf.Tensor,
                           target_db: float = 15.0,
                           max_gain_db: float = 30.0):
    """Normalizes the loudness of the input signal."""
    std = tf.math.reduce_std(samples) + 1e-9
    gain = tf.minimum(db_to_linear(max_gain_db), db_to_linear(target_db) / std)
    return gain * samples

def align(waveform: tf.Tensor, seq_len: int=16000 * 30):
    """make sure all training samples are the same length of time
    for shorter samples, repeat the same waveform again, then random crop
    """
    waveform_len = tf.size(waveform)
    pad_length = tf.maximum(seq_len - tf.size(waveform), 0)
    print(pad_length)
    pad_repeats = tf.cast(tf.math.ceil(pad_length / waveform_len), tf.int32)
    print(pad_repeats)
    samples = tf.squeeze(tf.tile(waveform, [pad_repeats + 1]))

    samples = tf.image.random_crop(tf.reshape(samples, (-1, 1)), [seq_len, 1])
    return tf.squeeze(samples)

def denoise(samples: tf.Tensor, threshold: float=0.1):
    mask = tf.greater(tf.abs(samples), threshold)
    samples = tf.where(mask, samples, tf.zeros_like(samples))
    return samples


@tf.function
def load_wav_16k_mono(filename):
    """ read in a waveform file and convert to 16 kHz mono """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
        file_contents,
        desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


def load_wav_for_map(filename, label, fold, transforms=None):
    wav = load_wav_16k_mono(filename)
    for fn in transforms:
        wav = fn(wav)
    return wav, label, fold


