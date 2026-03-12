import mlflow
from src.config import settings

def init_mlflow():
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)

class RunLogger:
    def __init__(self):
        init_mlflow()
        self.run = mlflow.start_run()

    def log_params(self, **params):
        for k, v in params.items():
            mlflow.log_param(k, v)

    def log_metrics(self, **metrics):
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))

    def log_text(self, text: str, artifact_file: str):
        mlflow.log_text(text, artifact_file)

    def end(self):
        mlflow.end_run()