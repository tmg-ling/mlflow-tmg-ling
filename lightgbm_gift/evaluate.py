import os

import mlflow
import mlflow.lightgbm
import pandas as pd


def feature_encoder(training_data, encoding_features):
    feature_mappings = {}
    for c in encoding_features:
        temp = training_data[c].astype("category").cat
        training_data[c] = temp.codes + 1
        feature_mappings[c] = {cat: n for n, cat in enumerate(temp.categories, start = 1)}
    return training_data, feature_mappings


def main():
    # prepare train and test data
    local_file = "../csv/65cb05a3-e45a-4a15-915b-90cf082dc203.csv"
    if not os.path.exists(local_file) and not os.path.isfile(local_file):
        filename = "s3://tmg-machine-learning-models-dev/for-you-payer-training-data/65cb05a3-e45a-4a15-915b-90cf082dc203.csv"
    else:
        filename = local_file

    df = pd.read_csv(filename)

    # prediction
    loaded_model = mlflow.pyfunc.load_model('mlruns/1/ec2190f63f2b48c7aea05d6a63685c00/artifacts/model')
    FEATURES = ["broadcaster_id", "viewer_id", "product_name", "ordered_time"]
    df, feature_mappings = feature_encoder(df, FEATURES)
    df["weight"] = 1
    print(df[:10])
    pred = loaded_model.predict(df[:10])
    print(pred)


if __name__ == "__main__":
    main()
