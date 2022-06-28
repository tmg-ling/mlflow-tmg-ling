[![MLflow testing build](https://github.com/tmg-ling/mlflow-tmg-ling/actions/workflows/main.yml/badge.svg)](https://github.com/tmg-ling/mlflow-tmg-ling/actions/workflows/main.yml)

![AWS Cloud build](https://codebuild.us-east-1.amazonaws.com/badges?uuid=eyJlbmNyeXB0ZWREYXRhIjoidkVqS2xWdGpvOHFCZ2hFd3BqalJoQ3gvT21GUXg1YjNxd0FFRFhyRStnSkVIT3dhNmloNksxVlNXTnBOSm8zVFQxdFFzbGNVSVZ2cHBVT3ZVb2tBOFlrPSIsIml2UGFyYW1ldGVyU3BlYyI6IjdhRnNJZ1pCN3BRKy92b0wiLCJtYXRlcmlhbFNldFNlcmlhbCI6MX0%3D&branch=main)

### About

MLflow is an open-source platform for managing ML lifecycles, including experimentation, deployment, and creation of a
central model registry. The MLflow Tracking component is an API that logs and loads the parameters, code versions, and
artifacts from ML model experiments.

### Setup

1. Start a virtual enviornment

```bash
python3 -m venv ~/.venv                  
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

- Train a model 
```bash
python lightgbm_gift/train.py --n_estimators 300 --learning_rate 1
python train_gift_dcm.py --experiment_name gift_model --batch_size 16384 --learning_rate 0.05
```

or run python in background

```bash
nohup python train_gift_dcm.py --experiment_name gift_model --batch_size 16384 --learning_rate 0.1 > nohup.out 2>&1 &
nohup python python train_gift_lightgbm.py --n_estimators 300 --learning_rate 1 > nohup.out 2>&1 &
```

4. Run mlflow

```bash
mlflow run .
mlflow ui
```

5. safely shut down
```
ps -A | grep gunicorn
```
Take the PID and kill the process

6. Run mlflow job

```
mlflow run  -P alpha=0.4
```

7. Build th docker image

```
docker build -t mlflow-tmg-ling .
```

9. Deploy the model in SageMaker using MLflow

```
experiment_id = 0
run_id = e820dfefbda4487b8abf6ecdce65d728

cd mlruns/0/e820dfefbda4487b8abf6ecdce65d728/artifacts/model
mlflow sagemaker build-and-push-container
```

```
yum update -y
yum install -y curl
http://localhost:5000
```

7. Start the serving API

8. Test the API
