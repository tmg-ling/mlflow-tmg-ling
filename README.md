[![MLflow testing build](https://github.com/tmg-ling/mlflow-tmg-ling/actions/workflows/main.yml/badge.svg)](https://github.com/tmg-ling/mlflow-tmg-ling/actions/workflows/main.yml)

![AWS Cloud build](https://codebuild.us-east-1.amazonaws.com/badges?uuid=eyJlbmNyeXB0ZWREYXRhIjoidkVqS2xWdGpvOHFCZ2hFd3BqalJoQ3gvT21GUXg1YjNxd0FFRFhyRStnSkVIT3dhNmloNksxVlNXTnBOSm8zVFQxdFFzbGNVSVZ2cHBVT3ZVb2tBOFlrPSIsIml2UGFyYW1ldGVyU3BlYyI6IjdhRnNJZ1pCN3BRKy92b0wiLCJtYXRlcmlhbFNldFNlcmlhbCI6MX0%3D&branch=main)

### About

MLflow is an open-source platform for managing ML lifecycles, including experimentation, deployment, and creation of a
central model registry. The MLflow Tracking component is an API that logs and loads the parameters, code versions, and
artifacts from ML model experiments.

- Host a serverless MLflow server 
- Set s3 as artifact and backend stores
- Track SageMaker experiments with Mlflow
- Register models trained in SageMaker in the MLflow Model Registry
- Deploy an MLflow model into a SageMaker endpoint

### Setup

1. Start a virtual enviornment

```bash             
source ~/.venv/bin/activate 
```

2. Make install requirements

Need the following files to install requirement enviornment

- Makefile: Makefile
- requirements.txt: requirements.txt

```bash
make all
```

3. Run training jobs

* Lightgbm native

```bash
cd lightgbm_native
python train.py --n_estimators 300 --learning_rate 1
nohup python train.py --experiment_name gift_model --batch_size 16384 --learning_rate 0.1 > nohup.out 2>&1 &
```

* Lightgbm regression

```bash
cd lightgbm_gift
python train.py --n_estimators 300 --learning_rate 1
nohup python train.py --n_estimators 300 --learning_rate 1 > nohup.out 2>&1 &
```

* Two Tower Model

```bash
cd tfrs_two_tower_gift
python train.py --batch_size 16384 --learning_rate 0.05 --broadcaster_embedding_dimension 96 --viewer_embedding_dimension 96 --top_k 1000
nohup python train.py --batch_size 16384 --learning_rate 0.05 --broadcaster_embedding_dimension 96 --viewer_embedding_dimension 96 --top_k 1000 > nohup.out 2>&1 &
```

* Deep and Cross Network

```bash
cd tfrs_dcn_gift
python train.py --batch_size 16384 --learning_rate 0.05
nohup python train.py --batch_size 16384 --learning_rate 0.05 > nohup.out 2>&1 &
```

* Listwise ranking

```bash
cd tfrs_listwise_ranking_gift
python train.py --batch_size 16384 --learning_rate 0.05
nohup python train.py --batch_size 16384 --learning_rate 0.05 > nohup.out 2>&1 &
```

4. Run mlflow

```bash
mlflow run .
mlflow run . -P learning_rate=0.01 -P n_estimators=300 
mlflow run . -P learning_rate=0.01 -P n_estimators=300 --experiment-name Baseline_Predictions
mlflow server --backend-store-uri sqlite:///:memory --default-artifact-root ./mlruns
```

5. check model results and safely shut down

```
mlflow ui
ps -A | grep gunicorn
```

Take the PID and kill the process

6.Build th docker image

```
docker build -t mlflow-tmg-ling .
```

7. Push the local image to ECR

```
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 882748442234.dkr.ecr.us-east-1.amazonaws.com
docker tag mlflow-tmg-ling:latest 882748442234.dkr.ecr.us-east-1.amazonaws.com/mlflow-pyfunc:latest

experiment_id = 0
run_id = e820dfefbda4487b8abf6ecdce65d728
cd mlruns/0/e820dfefbda4487b8abf6ecdce65d728/artifacts/model

mlflow sagemaker build-and-push-container
aws ecr describe-images --repository-name mlflow-pyfunc
docker push 882748442234.dkr.ecr.us-east-1.amazonaws.com/mlflow-pyfunc:latest
```

8. Deploy image to SageMaker

```
aws ecr describe-images --repository-name mlflow-pyfunc
python deploy.py
aws sagemaker list-endpoints
```

9. Evaluate the predictions

```
python evaluate.py
```

10. Predict with Athena and SageMaker endpoint

```
DROP TABLE gift_testing_data;

CREATE EXTERNAL TABLE gift_testing_data
    (
        `viewer_id` string,
        `broadcaster_id` string,
        `product_name` string,
        `ordered_time` string,
        `count` int
    )
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
   'separatorChar' = ',',
   'quoteChar' = '"',
   'escapeChar' = '\\'
   )
STORED AS TEXTFILE
LOCATION 's3://tmg-machine-learning-models-dev/for-you-payer-training-data/'
TBLPROPERTIES('skip.header.line.count'='1')
;
USING EXTERNAL FUNCTION predict_avg_gift (broadcaster_id VARCHAR, 
    viewer_id VARCHAR, 
    product_name VARCHAR, 
    ordered_time VARCHAR
) 
RETURNS DOUBLE 
SAGEMAKER 'lightgbm-gift'
SELECT 
    predict_avg_gift("broadcaster_id","viewer_id", "product_name", "ordered_time") AS prediction
FROM gift_testing_data
LIMIT 10
```
