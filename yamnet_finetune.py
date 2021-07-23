import tensorflow as tf
import tensorflow_hub as hub
import json
import pandas as pd
from data import loudness_normalization, align


from sklearn.model_selection import train_test_split
from functools import partial

from data import fetch_metadata_v2,load_wav_for_map, load_wav_16k_mono, bird_mapping_idx_to_name
from yamnet import Yamnet

if __name__ == '__main__':
    # tf.debugging.set_log_device_placement(True)
    batch_size = 64

    assert len(tf.config.list_physical_devices('GPU')) >= 1
    yamnet_base = Yamnet()
    yamnet_class_names = yamnet_base.yamnet_class_names

    birdsong_metadata, bird_mapping_idx_to_name = fetch_metadata_v2()
    dataset = birdsong_metadata.copy()

    train_audio_paths, valtest_audio_paths, train_labels, valtest_labels = train_test_split(
        dataset[["filepath"]], dataset["label"], stratify=dataset["label"], test_size=0.3, random_state=101
    )
    val_audio_paths, test_audio_paths, val_labels, test_labels = train_test_split(
        valtest_audio_paths, valtest_labels, stratify=valtest_labels, test_size=0.5, random_state=101
    )

    print(f"there are {len(train_labels)} train audios")
    print(f"there are {len(val_labels)} val audios")
    print(f"there are {len(test_labels)} test audios")

    filepaths = pd.concat([train_audio_paths, val_audio_paths])
    folds = pd.Series([0] * len(train_audio_paths) + [1] * len(val_audio_paths))
    targets = pd.concat([train_labels, val_labels])

    main_ds = tf.data.Dataset.from_tensor_slices(
        (filepaths.values.squeeze(), targets.values.squeeze(), folds.values.squeeze()))

    load_wav_fn = partial(load_wav_for_map, transforms=[align, loudness_normalization])
    main_ds = main_ds.map(load_wav_fn)

    main_ds = main_ds.map(yamnet_base.extract_embedding).unbatch()
    cached_ds = main_ds.cache()
    train_ds = cached_ds.filter(lambda embedding, label, fold: fold == 0)
    val_ds = cached_ds.filter(lambda embedding, label, fold: fold == 1)

    remove_fold_column = lambda embedding, label, fold: (embedding, label)

    train_ds = train_ds.map(remove_fold_column)
    val_ds = val_ds.map(remove_fold_column)

    train_ds = train_ds.cache().shuffle(1000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.cache().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    ym_finetune_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024), dtype=tf.float32,
                              name='input_embedding'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(birdsong_metadata.label.nunique())
    ], name='ym_finetune_model')

    metrics = [
        tf.keras.metrics.SparseTopKCategoricalAccuracy(
            k=5, name="sparse_top_5_categorical_accuracy", dtype=None
        ),
        'accuracy'
    ]
    optimiser = tf.keras.optimizers.Adam(
        learning_rate=5e-4,
    )
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    ym_finetune_model.compile(loss=loss,
                              optimizer=optimiser,
                              metrics=metrics)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath="test_model_all_birds_v2",
            save_best_only=True,  # Only save a model if `val_loss` has improved.
            monitor="val_sparse_top_5_categorical_accuracy",
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir="./logs",
            update_freq="batch",
        ),
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
    history = ym_finetune_model.fit(train_ds,
                                    epochs=30,
                                    validation_data=val_ds,
                                    callbacks=callbacks)
    ym_finetune_model.save('/media/wwymak/Storage2/birdsong_dataset/models/ym_finetune_baseline_transforms')
    with open('/media/wwymak/Storage2/birdsong_dataset/models/bird_mapping_idx_to_name.json', 'w') as f:
        json.dump(bird_mapping_idx_to_name, f)