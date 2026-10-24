import os
from urllib.parse import urlparse

import mlflow
import requests

from ml_network_security_system.logger.log import logging

from ml_network_security_system.constant.training_pipeline import (
    MLFLOW_DEFAULT_SERVER,
    MLFLOW_OFFLINE_URI,
    MLFLOW_EXPERIMENT_NAME,
)

def _server_alive(uri: str) -> bool:
    try:
        return requests.get(f"{uri.rstrip('/')}/health", timeout=2).ok
    except requests.RequestException:
        return False


def setup_mlflow(experiment_name: str | None = None) -> str:
    """
    Configure MLflow tracking once.
    - MLFLOW_TRACKING_URI env var wins (e.g. DagsHub URL); default is the local server.
    - If the local server is not running, fall back to an offline SQLite store
      so the pipeline never fails because of MLflow.
    """
    uri = os.getenv("MLFLOW_TRACKING_URI", MLFLOW_DEFAULT_SERVER)

    is_local_server = urlparse(uri).hostname in ("127.0.0.1", "localhost")
    if is_local_server and not _server_alive(uri):
        logging.warning(
            f"MLflow server not reachable at {uri}. Logging offline to {MLFLOW_OFFLINE_URI} "
            f"(view with: mlflow ui --backend-store-uri {MLFLOW_OFFLINE_URI} --port 5001)"
        )
        uri = MLFLOW_OFFLINE_URI

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME", MLFLOW_EXPERIMENT_NAME))
    logging.info(f"MLflow tracking URI: {uri}")
    return uri