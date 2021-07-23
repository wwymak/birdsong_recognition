# coding=utf-8
# Copyright 2021 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training loop using the LEAF frontend."""

import os
from typing import Optional

import gin
from leaf_audio import models
from data import get_dataset
import data
import tensorflow as tf
import tensorflow_datasets as tfds


@gin.configurable
def train(
    dataset_dir: str,
    model_dir: str = '/media/wwymak/Storage2/birdsong_dataset/models',
            num_classes: int = 84,
          num_epochs: int = 10,
          steps_per_epoch: Optional[int] = None,
          learning_rate: float = 1e-4,
          batch_size: int = 64,
          **kwargs):
  """Trains a model on a dataset.

  Args:
    workdir: where to store the checkpoints and metrics.
    dataset: name of a tensorflow_datasets audio datasset.
    num_epochs: number of epochs to training the model for.
    steps_per_epoch: number of steps that define an epoch. If None, an epoch is
      a pass over the entire training set.
    learning_rate: Adam's learning rate.
    batch_size: size of the mini-batches.
    **kwargs: arguments to the models.AudioClassifier class, namely the encoder
      and the frontend models (tf.keras.Model).
  """
  datasets = data.get_dataset(dataset_dir)
  datasets = data.prepare(datasets, batch_size=batch_size)
  model = models.AudioClassifier(num_outputs=num_classes, **kwargs)
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  metric = 'sparse_categorical_accuracy'


  ckpt_path = os.path.join(model_dir, 'leaf_checkpoint')

  callbacks = [
      tf.keras.callbacks.ModelCheckpoint(
          filepath=ckpt_path,
          save_weights_only=True,
          monitor=f'val_sparse_top_5_categorical_accuracy',
          mode='max',
          save_best_only=True),
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

  metrics = [
      tf.keras.metrics.SparseTopKCategoricalAccuracy(
          k=5, name="sparse_top_5_categorical_accuracy", dtype=None
      ),
      'sparse_categorical_accuracy'
  ]

  model.compile(loss=loss_fn,
                optimizer=tf.keras.optimizers.Adam(learning_rate),
                metrics=metrics)


  model.fit(datasets['train'],
            validation_data=datasets['eval'],
            batch_size=None,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks)
