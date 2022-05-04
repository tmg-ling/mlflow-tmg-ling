[![MLops testing build](https://github.com/tmg-ling/mlops-tmg-ling/actions/workflows/main.yml/badge.svg)](https://github.com/tmg-ling/mlops-tmg-ling/actions/workflows/main.yml)

![AWS Cloud build](https://codebuild.us-east-1.amazonaws.com/badges?uuid=eyJlbmNyeXB0ZWREYXRhIjoidkVqS2xWdGpvOHFCZ2hFd3BqalJoQ3gvT21GUXg1YjNxd0FFRFhyRStnSkVIT3dhNmloNksxVlNXTnBOSm8zVFQxdFFzbGNVSVZ2cHBVT3ZVb2tBOFlrPSIsIml2UGFyYW1ldGVyU3BlYyI6IjdhRnNJZ1pCN3BRKy92b0wiLCJtYXRlcmlhbFNldFNlcmlhbCI6MX0%3D&branch=main)


MLflow is an open-source platform for managing ML lifecycles, including experimentation, deployment, and creation of a central model registry.
The MLflow Tracking component is an API that logs and loads the parameters, code versions, and artifacts from ML model experiments.
- mlflow.tensorflow.autolog() enables you to automatically log the experiment in the local directory. It captures the metrics produced by the underlying ML library in use. MLflow Tracking is the module responsible for handling metrics and logs. By default, the metadata of an MLflow run is stored in the local filesystem.
  - The MLmodel file is the main definition of the project from an MLflow project with information related to how to run inference on the current model.
  - The metrics folder contains the training score value of this particular run of the training process, which can be used to benchmark the model with further model improvements down the line.
  - The params folder on the first listing of folders contains the default parameters of the logistic regression model, with the different default possibilities listed transparently and stored automatically.


  
` ``bash
  python -m venv ~/.venv                  
 source ~/.venv/bin/activate  # mlops-tmg-ling
 ```

- Makefile: View Makefile
- requirements.txt: View requirements.txt

[comment]: <> (cli.py: View cli.py)

[comment]: <> (utilscli.py: View utilscli.py)

[comment]: <> (app.py: View app.py)

[comment]: <> (mlib.py: View mlib.pyModel Handling Library)

[comment]: <> (htwtmlb.csv: View CSV Useful for input scaling)

[comment]: <> (model.joblib: View model.joblib)

[comment]: <> (Dockerfile: View Dockerfile)

[comment]: <> (notbooks/*.ipynb)

```
mlflow ui
``
