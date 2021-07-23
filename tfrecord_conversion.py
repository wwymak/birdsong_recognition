import soundfile as sf
import tensorflow as tf
from pathlib import Path
import numpy as np
from tqdm import tqdm

from data import fetch_metadata_v2
from sklearn.model_selection import train_test_split


def audio_feature(waveform_np):
    """Returns a float_list from a audio np array"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=waveform_np))

def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

def int64_feature(value):
    """Create a Int64List Feature

    Args:
        value: The value to store in the feature

    Returns:
        The FeatureEntry
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def create_example(waveform, filepath, metadata):
    feature = {
        "audio": audio_feature(waveform),
        "filepath": bytes_feature(filepath),
        "section": int64_feature(metadata["section"]),
        "category_id": int64_feature(metadata["category_id"]),
        "audio_id": int64_feature(metadata["audio_id"]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def create_tf_records(num_tf_records, tfrecords_dir, prefix, audio_filepaths, labels, sample_rate=16_000, seconds_per_chunk=10):
    num_audios_per_record = len(audio_filepaths) // num_tf_records
    for tfrec_num in range(num_tf_records):
        with tf.io.TFRecordWriter(
                str(tfrecords_dir / f"file_{tfrec_num}_{prefix}.tfrec")
        ) as writer:
            for audio_filepath, label in tqdm(list(
                    zip(audio_filepaths[tfrec_num *num_audios_per_record: (tfrec_num + 1) * num_audios_per_record], labels[tfrec_num *num_audios_per_record: (tfrec_num + 1) * num_audios_per_record]))):
                if not Path(audio_filepath).exists():
                    print(f'error: {audio_filepath}, {label}')
                    continue
                metadata = dict(category_id=label, filepath=audio_filepath,
                                audio_id=int(audio_filepath.split('/')[-1].replace('.wav', '')))
                audio_np = sf.read(audio_filepath)[0]
                num_chunks = np.ceil(len(audio_np) / (sample_rate * seconds_per_chunk)).astype(int)
                for i in range(num_chunks):
                    metadata['section'] = i

                    audio_array = audio_np[
                                  i * (sample_rate * seconds_per_chunk): (i + 1) * (sample_rate * seconds_per_chunk)]

                    if len(audio_array) < (sample_rate * seconds_per_chunk):
                        pad_times = np.ceil((sample_rate * seconds_per_chunk) / len(audio_array)).astype(int)
                        audio_array = np.tile(audio_array, pad_times)
                        audio_array = audio_array[:(sample_rate * seconds_per_chunk)]
                    example = create_example(audio_array, audio_filepath, metadata)
                    writer.write(example.SerializeToString())

if __name__ == '__main__':
    birdsong_metadata = fetch_metadata_v2()
    dataset = birdsong_metadata.copy()
    label_counts = dataset.label.value_counts

    train_audio_paths, valtest_audio_paths, train_labels, valtest_labels = train_test_split(
        dataset[["filepath"]], dataset["label"], stratify=dataset["label"], test_size=0.3, random_state=101
    )
    val_audio_paths, test_audio_paths, val_labels, test_labels = train_test_split(
        valtest_audio_paths, valtest_labels, stratify=valtest_labels, test_size=0.5, random_state=101
    )

    print(f"there are {len(train_labels)} train audios")
    print(f"there are {len(val_labels)} val audios")
    print(f"there are {len(test_labels)} test audios")

    tfrecords_dir = Path('/media/wwymak/Storage2/birdsong_dataset/xeno_canto_eu_cleaned_stage2_tfrecord')
    tfrecords_dir.mkdir(exist_ok=True)

    num_tfrecords_train = 10
    num_tfrecords_val = 1
    num_tfrecords_test = 1

    sample_rate = 16_000
    seconds_per_chunk = 10

    create_tf_records(num_tfrecords_train, tfrecords_dir, 'train', train_audio_paths.values.squeeze(),
                      train_labels.astype(int).values, sample_rate=16_000, seconds_per_chunk=10)
    create_tf_records(num_tfrecords_val, tfrecords_dir, 'val', val_audio_paths.values.squeeze(),
                      val_labels.astype(int).values, sample_rate=16_000, seconds_per_chunk=10)
    create_tf_records(num_tfrecords_test, tfrecords_dir, 'test', test_audio_paths.values.squeeze(),
                      test_labels.astype(int).values, sample_rate=16_000, seconds_per_chunk=10)

