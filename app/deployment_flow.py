import mlflow
from fire import Fire
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = 'first_experiment'
try:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
except:
    mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)


class HandleModel:
    def __init__(self, model_name: str, version: int = 1):
        self.model_name = model_name
        self.version = version
        self.client = MlflowClient()

    def register_model(self, run_id: str) -> None:
        mlflow.register_model(f"runs:/{run_id}/model", self.model_name)

    def update_meta(self, description) -> None:
        self.client.update_model_version(
            name=self.model_name,
            version=self.version,
            description=description,
        )

    def assign_alias(self, alias: str = 'staging') -> None:
        self.client.set_registered_model_alias(
            name=self.model_name, alias=alias, version=f"{self.version}"
        )

    def tag_model(self, key: str = 'env', value: str = 'staging') -> None:
        self.client.set_model_version_tag(
            name=self.model_name, version=f"{self.version}", key=key, value=value
        )


client = MlflowClient()
aliases = client.get_model_version_by_alias(name="first_lstm", alias='staging')
print(aliases)

print(mlflow.get_tracking_uri())

if __name__ == "__main__":
    Fire({"HandleModel": HandleModel})
