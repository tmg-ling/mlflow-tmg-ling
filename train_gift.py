# in case this is run outside of conda environment with python2
import argparse
import logging
import os
import sys
import time
from collections import defaultdict

import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np
import pandas as pd

import mlflow


class DCN(tfrs.Model):
    def __init__(self, conf, use_cross_layer, deep_layer_sizes, projection_dim=None):
        super().__init__()

        self.embedding_dimension = 32

        str_features = ["broadcaster", "viewer", "product_name", "order_time"]
        int_features = []

        self._all_features = str_features
        self._embeddings = {}

        # Compute embeddings for string features.
        vocabularies = conf["vocabularies"]
        for feature_name in str_features:
            vocabulary = vocabularies[feature_name]
            self._embeddings[feature_name] = tf.keras.Sequential(
                [
                    tf.keras.layers.experimental.preprocessing.StringLookup(
                        vocabulary=vocabulary, mask_token=None
                    ),
                    tf.keras.layers.Embedding(
                        len(vocabulary) + 1, self.embedding_dimension
                    ),
                ]
            )

        # Compute embeddings for int features.
        for feature_name in int_features:
            vocabulary = vocabularies[feature_name]
            self._embeddings[feature_name] = tf.keras.Sequential(
                [
                    tf.keras.layers.IntegerLookup(
                        vocabulary=vocabulary, mask_value=None
                    ),
                    tf.keras.layers.Embedding(
                        len(vocabulary) + 1, self.embedding_dimension
                    ),
                ]
            )

        if use_cross_layer:
            self._cross_layer = tfrs.layers.dcn.Cross(
                projection_dim=projection_dim, kernel_initializer="glorot_uniform"
            )
        else:
            self._cross_layer = None

        self._deep_layers = [
            tf.keras.layers.Dense(layer_size, activation="relu")
            for layer_size in deep_layer_sizes
        ]

        self._logit_layer = tf.keras.layers.Dense(1)

        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError("RMSE")],
        )

    def call(self, features):
        # Concatenate embeddings
        embeddings = []
        for feature_name in self._all_features:
            embedding_fn = self._embeddings[feature_name]
            embeddings.append(embedding_fn(features[feature_name]))

        x = tf.concat(embeddings, axis=1)

        # Build Cross Network
        if self._cross_layer is not None:
            x = self._cross_layer(x)

        # Build Deep Network
        for deep_layer in self._deep_layers:
            x = deep_layer(x)

        return self._logit_layer(x)

    def compute_loss(self, features, training=False):
        labels = features.pop("count")
        scores = self(features)
        return self.task(
            labels=labels,
            predictions=scores,
        )


def load_data_file_gift(file, stats):
    print("loading file:" + file)
    training_df = pd.read_csv(
        file,
        skiprows=[0],
        names=["broadcaster", "viewer", "product_name", "order_time", "count"],
        dtype={
            "broadcaster": np.unicode,
            "viewer": np.unicode,
            "product_name": np.unicode,
            "order_time": np.unicode,
            "count": np.int,
        },
    )

    values = {
        "broadcaster": "unknown",
        "viewer": "unknown",
        "product_name": "unknown",
        "order_time": "0",
        "count": "0",
    }
    training_df = training_df.sample(n=1000)
    training_df.fillna(value=values, inplace=True)
    return training_df


def load_training_gift(file, stats):
    df = load_data_file_gift(file, stats)
    print("creating data set")
    training_ds = tf.data.Dataset.from_tensor_slices(
        (
            {
                "viewer": tf.cast(df["viewer"].values, tf.string),
                "broadcaster": tf.cast(df["broadcaster"].values, tf.string),
                "product_name": tf.cast(df["product_name"].values, tf.string),
                "order_time": tf.cast(df["order_time"].values, tf.string),
                "count": tf.cast(df["count"].values, tf.int64),
            }
        )
    )
    return training_ds, len(df)


def prepare_training_data_gift(train_ds):
    print("prepare_training_data")
    training_ds = train_ds.map(
        lambda x: {
            "broadcaster": x["broadcaster"],
            "viewer": x["viewer"],
            "product_name": x["product_name"],
            "order_time": x["order_time"],
            "count": x["count"],
        },
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    return training_ds


def main(argv):
    conf = defaultdict(dict)
    conf["broadcaster_embedding_dimension "] = 96
    conf["viewer_embedding_dimension"] = 96
    conf["batch_size"] = 16384
    conf["learning_rate"] = 0.05
    conf["epochs"] = 3
    conf["top_k"] = 1000

    # Fetch the data
    dataset, nrow = load_training_gift(
        "csv/f880878d-4944-40f8-b80f-f47b5841398c.csv", ""
    )
    gift = prepare_training_data_gift(dataset)
    shuffled = gift.shuffle(nrow, seed=42, reshuffle_each_iteration=False)

    ds_train = shuffled.take(int(nrow * 0.8))
    ds_train = ds_train.cache()
    ds_train = ds_train.batch(conf["batch_size"])
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = shuffled.skip(int(nrow * 0.8)).take(int(nrow * 0.2))
    ds_test = ds_test.batch(conf["batch_size"])
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    # Fetch feature and vocabularies
    feature_names = ["broadcaster", "viewer", "product_name", "order_time"]
    vocabularies = {}
    for idx, feature_name in enumerate(feature_names):
        print(f"{idx}: {feature_name}")
        vocab = gift.batch(1_000_000).map(
            lambda x: x[feature_name],
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )
        vocabularies[feature_name] = np.unique(np.concatenate(list(vocab)))
    conf["vocabularies"] = vocabularies

    # Train the Model.
    model = DCN(
        conf=conf,
        use_cross_layer=True,
        deep_layer_sizes=[192, 192],
        projection_dim=None,
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(conf["learning_rate"]))
    model.fit(ds_train, epochs=conf["epochs"], verbose=False)
    metrics = model.evaluate(ds_test, return_dict=True)
    # tf.keras.models.save_model(model, "./model")

    with mlflow.start_run(run_name="Gift Model Experiments") as run:
        run_id = run.info.run_uuid
        experiment_id = run.info.experiment_id
        mlflow.log_param("batch_size", conf["batch_size"])
        mlflow.log_param("learning_rate", conf["learning_rate"])
        mlflow.log_param("epochs", conf["epochs"])
        mlflow.log_metric("RMSE", metrics["RMSE"])
        mlflow.log_artifacts("./model")
        print(f"runid: {run_id}")
        print(f"experimentid: {experiment_id}")


if __name__ == "__main__":
    main(sys.argv)
