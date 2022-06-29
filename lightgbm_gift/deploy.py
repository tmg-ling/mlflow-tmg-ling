import mlflow.sagemaker as mfs

experiment_id = '0'
run_id = '0089b8a97b244cc19aabe7006f21508a'
region = 'us-east-1'
aws_id = '882748442234'
arn = 'arn:aws:iam::882748442234:role/service-role/AmazonSageMaker-ExecutionRole-20210915T104260'

app_name = 'lightgbm-gift'
model_uri = 'mlruns/%s/%s/artifacts/model' % (experiment_id, run_id)
tag_id = 'latest'

image_url = aws_id + '.dkr.ecr.' + region + '.amazonaws.com/mlflow-pyfunc:' + tag_id

mfs.deploy(app_name=app_name,
           model_uri=model_uri,
           region_name=region,
           mode="create",
           execution_role_arn=arn,
           image_url=image_url)