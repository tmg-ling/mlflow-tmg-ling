# in case this is run outside of conda environment with python2
import argparse
import mlflow
import mlflow.tensorflow
import pandas as pd
import shutil
import sys
import tempfile
import tensorflow as tf
from mlflow import pyfunc
from tensorflow import estimator as tf_estimator

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Species"]
SPECIES = ["Setosa", "Versicolor", "Virginica"]


def load_data(y_name = "Species"):
	"""Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
	train_path = tf.keras.utils.get_file(TRAIN_URL.split("/")[-1], TRAIN_URL)
	test_path = tf.keras.utils.get_file(TEST_URL.split("/")[-1], TEST_URL)

	train = pd.read_csv(train_path, names = CSV_COLUMN_NAMES, header = 0)
	train_x, train_y = train, train.pop(y_name)

	test = pd.read_csv(test_path, names = CSV_COLUMN_NAMES, header = 0)
	test_x, test_y = test, test.pop(y_name)

	return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
	"""An input function for training"""
	# Convert the inputs to a Dataset.
	dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

	# Shuffle, repeat, and batch the examples.
	dataset = dataset.shuffle(1000).repeat().batch(batch_size)

	# Return the dataset.
	return dataset


def eval_input_fn(features, labels, batch_size):
	"""An input function for evaluation or prediction"""
	features = dict(features)
	if labels is None:
		# No labels, use only features.
		inputs = features
	else:
		inputs = (features, labels)

	# Convert the inputs to a Dataset.
	dataset = tf.data.Dataset.from_tensor_slices(inputs)

	# Batch the examples
	assert batch_size is not None, "batch_size must not be None"
	dataset = dataset.batch(batch_size)

	# Return the dataset.
	return dataset


# Enable auto-logging to MLflow to capture TensorBoard metrics.
mlflow.tensorflow.autolog()

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default = 100, type = int, help = "batch size")
parser.add_argument("--train_steps", default = 1000, type = int, help = "number of training steps")


def main(argv):
	with mlflow.start_run():
		args = parser.parse_args(argv[1:])

		# Fetch the data
		(train_x, train_y), (test_x, test_y) = load_data()

		# Feature columns describe how to use the input.
		my_feature_columns = []
		for key in train_x.keys():
			my_feature_columns.append(tf.feature_column.numeric_column(key = key))

		# Two hidden layers of 10 nodes each.
		hidden_units = [10, 10]

		# Build 2 hidden layer DNN with 10, 10 units respectively.
		classifier = tf_estimator.DNNClassifier(
			feature_columns = my_feature_columns,
			hidden_units = hidden_units,
			# The model must choose between 3 classes.
			n_classes = 3,
		)

		# Train the Model.
		classifier.train(
			input_fn = lambda: train_input_fn(train_x, train_y, args.batch_size),
			steps = args.train_steps,
		)

		# Evaluate the model.
		eval_result = classifier.evaluate(
			input_fn = lambda: eval_input_fn(test_x, test_y, args.batch_size)
		)

		print("\nTest set accuracy: {accuracy:0.3f}\n".format(**eval_result))

		# Generate predictions from the model
		expected = ["Setosa", "Versicolor", "Virginica"]
		predict_x = {
			"SepalLength": [5.1, 5.9, 6.9],
			"SepalWidth": [3.3, 3.0, 3.1],
			"PetalLength": [1.7, 4.2, 5.4],
			"PetalWidth": [0.5, 1.5, 2.1],
		}

		predictions = classifier.predict(
			input_fn = lambda: eval_input_fn(predict_x, labels = None, batch_size = args.batch_size)
		)

		old_predictions = []
		template = '\nPrediction is "{}" ({:.1f}%), expected "{}"'

		for pred_dict, expec in zip(predictions, expected):
			class_id = pred_dict["class_ids"][0]
			probability = pred_dict["probabilities"][class_id]

			print(template.format(SPECIES[class_id], 100 * probability, expec))

			old_predictions.append(SPECIES[class_id])

		# Creating output tf.Variables to specify the output of the saved model.
		feat_specifications = {
			"SepalLength": tf.Variable([], dtype = tf.float64, name = "SepalLength"),
			"SepalWidth": tf.Variable([], dtype = tf.float64, name = "SepalWidth"),
			"PetalLength": tf.Variable([], dtype = tf.float64, name = "PetalLength"),
			"PetalWidth": tf.Variable([], dtype = tf.float64, name = "PetalWidth"),
		}

		receiver_fn = tf_estimator.export.build_raw_serving_input_receiver_fn(feat_specifications)
		temp = tempfile.mkdtemp()
		try:
			# The model is automatically logged when export_saved_model() is called.
			saved_estimator_path = classifier.export_saved_model(temp, receiver_fn).decode("utf-8")

			# Since the model was automatically logged as an artifact (more specifically
			# a MLflow Model), we don't need to use saved_estimator_path to load back the model.
			# MLflow takes care of it!
			pyfunc_model = pyfunc.load_model(mlflow.get_artifact_uri("model"))

			predict_data = [[5.1, 3.3, 1.7, 0.5], [5.9, 3.0, 4.2, 1.5], [6.9, 3.1, 5.4, 2.1]]
			df = pd.DataFrame(
				data = predict_data,
				columns = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"],
			)

			# Predicting on the loaded Python Function and a DataFrame containing the
			# original data we predicted on.
			predict_df = pyfunc_model.predict(df)

			# Checking the PyFunc's predictions are the same as the original model's predictions.
			template = '\nOriginal prediction is "{}", reloaded prediction is "{}"'
			for expec, pred in zip(old_predictions, predict_df["classes"]):
				class_id = predict_df["class_ids"][
					predict_df.loc[predict_df["classes"] == pred].index[0]
				]
				reloaded_label = SPECIES[class_id]
				print(template.format(expec, reloaded_label))
		finally:
			shutil.rmtree(temp)


if __name__ == "__main__":
	main(sys.argv)

import tensorflow as tf
import tensorflow_datasets as tfds
import mlflow

# code paritally adapted from https://www.tensorflow.org/datasets/keras_example

batch_size = 128
learning_rate = 0.001
epochs = 10

(ds_train, ds_test), ds_info = tfds.load(
	'mnist',
	split = ['train', 'test'],
	shuffle_files = True,
	as_supervised = True,
	with_info = True,
)


def normalize_img(image, label):
	"""Normalizes images: `uint8` -> `float32`."""
	return tf.cast(image, tf.float32) / 255., label


ds_train = ds_train.map(
	normalize_img, num_parallel_calls = tf.data.experimental.AUTOTUNE
)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(batch_size)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(
	normalize_img, num_parallel_calls = tf.data.experimental.AUTOTUNE
)
ds_test = ds_test.batch(batch_size)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

model = tf.keras.models.Sequential(
	[
		tf.keras.layers.Flatten(input_shape = (28, 28)),
		tf.keras.layers.Dense(128, activation = 'relu'),
		tf.keras.layers.Dense(10)
	]
)
model.compile(
	optimizer = tf.keras.optimizers.Adam(learning_rate),
	loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
	metrics = [tf.keras.metrics.SparseCategoricalAccuracy()],
)

history = model.fit(
	ds_train,
	epochs = epochs,
	validation_data = ds_test,
)

train_loss = history.history['loss'][-1]
train_acc = history.history['sparse_categorical_accuracy'][-1]
val_loss = history.history['val_loss'][-1]
val_acc = history.history['val_sparse_categorical_accuracy'][-1]

print("train_loss: ", train_loss)
print("train_accuracy: ", train_acc)
print("val_loss: ", val_loss)
print("val_accuracy: ", val_acc)

tf.keras.models.save_model(model, "./model")

run_name = "firstRun"

with mlflow.start_run(run_name = run_name):
	mlflow.log_param("batch_size", batch_size)
	mlflow.log_param("learning_rate", learning_rate)
	mlflow.log_param("epochs", epochs)
	mlflow.log_metric("train_loss", train_loss)
	mlflow.log_metric("train_accuracy", train_acc)
	mlflow.log_metric("val_loss", val_loss)
	mlflow.log_metric("val_accuracy", val_acc)
	mlflow.log_artifacts("./model")
