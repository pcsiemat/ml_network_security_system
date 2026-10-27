"""
=====================================================================
 MODEL TRAINER - Stage 4 of the training pipeline
=====================================================================
INPUT  : DataTransformationArtifact (train.npy, test.npy, preprocessing.pkl)
         ModelTrainerConfig         (where to save the trained model)
STEPS  : 1. load train.npy / test.npy → split X (all columns but last) and y (last column)
         2. tune + train 5 classifiers with GridSearchCV (scoring = f1, phishing = 1)
         3. pick the model with the best CROSS-VALIDATION f1 (train data only)
         4. checks on the chosen model:
              - underfitting : test f1 >= expected_accuracy                       (else stop)
              - overfitting  : train f1 - test f1 <= overfitting_underfitting_threshold (else stop)
              - stability    : std of f1 across CV folds <= CV_STD_WARNING        (else warning)
         5. compute precision / recall / f1 on train and test
            + track in MLflow (one run per training):
              - winner    : params + metrics + model registered under a FIXED name
              - champion  : alias moves only if the new version is at least as good
              - candidates: metrics prefixed with the model name + model_report.json
         6. wrap preprocessor + model in NetworkModel and save it as model.pkl (Artifacts history)
         7. promote to final_model/ (preprocessor + model + metrics) only if better than production
OUTPUT : ModelTrainerArtifact
=====================================================================
"""

import os
import sys
from dataclasses import asdict

import matplotlib
matplotlib.use("Agg")          # no window, just save the image
import matplotlib.pyplot as plt
import pandas as pd


import dagshub
import mlflow
import skops.io as sio
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from ml_network_security_system.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from ml_network_security_system.entity.config_entity import ModelTrainerConfig
from ml_network_security_system.exception.exception import AppException
from ml_network_security_system.logger.log import logging
from ml_network_security_system.utils.main_utils.utils import (
    evaluate_models,
    load_numpy_array_data,
    load_object,
    read_yaml_file,
    save_object,
    write_yaml_file,
)
from ml_network_security_system.utils.main_utils.mlflow_utils import setup_mlflow
from ml_network_security_system.utils.ml_utils.metric.classification_metric import get_classification_score
from ml_network_security_system.utils.ml_utils.model.estimator import NetworkModel


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

            try:
                dagshub.init(
                    repo_owner=model_trainer_config.dagshub_repo_owner,
                    repo_name=model_trainer_config.dagshub_repo_name,
                    mlflow=True,
                )
            except Exception as e:
                logging.warning(f"DagsHub not available ({e}). Falling back to local MLflow / offline.")
            setup_mlflow()
        except Exception as e:
            raise AppException(e, sys)

    # ------------------------------------------------------------------ MLflow
    def _update_champion(self, model_name: str, new_version: str, new_f1: float) -> None:
        """Move the 'champion' alias to the new version only if it is at least as good."""
        client = MlflowClient()
        try:
            champion = client.get_model_version_by_alias(model_name, "champion")
            champion_f1 = client.get_run(champion.run_id).data.metrics.get("test_f1_score")
        except MlflowException:
            champion, champion_f1 = None, None   # no champion yet

        if champion_f1 is None or new_f1 >= champion_f1:
            client.set_registered_model_alias(model_name, "champion", new_version)
            logging.info(f"champion -> {model_name} v{new_version} (test f1 {new_f1} vs {champion_f1})")
        else:
            logging.info(f"champion stays at v{champion.version}: new test f1 {new_f1} < {champion_f1}")

    def _candidates_figure(self, model_report: dict, best_model_name: str):
        """Grouped bar chart: cv / train / test f1 of every candidate."""
        df = pd.DataFrame(model_report).T[["cv_f1_mean", "train_f1", "test_f1"]]
        ax = df.plot(kind="bar", figsize=(10, 5), ylim=(max(0, df.values.min() - 0.05), 1))
        ax.set_title(f"Candidates comparison (best: {best_model_name})")
        ax.set_ylabel("f1")
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=20)
        plt.tight_layout()
        return ax.figure


    def track_mlflow(self, best_model_name, best_model, model_report, train_metric, test_metric):
        """
        One MLflow parent run per training, with one nested run per candidate:
        - parent    : winner params + train/test metrics + model registered under a FIXED name
                      + model_report.json + candidates_comparison.png
        - champion  : alias moves only if the new version is at least as good
        - children  : metrics of every candidate (grouped under the parent in the MLflow UI)
        Filter parents only with: tags.run_type = "training"
        """
        registered_name = self.model_trainer_config.registered_model_name

        # Types that skops flags as "untrusted" for THIS specific model.
        # Safe to trust them because we just trained this model ourselves.
        trusted_types = sio.get_untrusted_types(data=sio.dumps(best_model))

        with mlflow.start_run(run_name=f"training - {best_model_name}"):
            ## winner
            mlflow.set_tag("run_type", "training")
            mlflow.set_tag("algorithm", best_model_name)
            mlflow.log_params(best_model.get_params())

            # train_f1_score, test_f1_score, train_precision_score, ...
            for prefix, metric in (("train", train_metric), ("test", test_metric)):
                mlflow.log_metrics({f"{prefix}_{k}": v for k, v in asdict(metric).items()})

            model_info = mlflow.sklearn.log_model(
                best_model,
                name="model",
                skops_trusted_types=trusted_types,
                registered_model_name=registered_name,
            )

            ## champion alias
            try:
                self._update_champion(registered_name, model_info.registered_model_version,
                                      float(test_metric.f1_score))
            except Exception as e:
                logging.warning(f"Could not update champion alias: {e}")

            ## candidates: one nested run each (grouped under the parent in the MLflow UI)
            for name, report in model_report.items():
                with mlflow.start_run(run_name=name, nested=True):
                    mlflow.set_tag("run_type", "candidate")
                    mlflow.set_tag("is_best", name == best_model_name)
                    mlflow.log_metrics({k: v for k, v in report.items() if isinstance(v, (int, float))})

            ## summary in the parent run (shows up in its Artifacts tab, also in DagsHub)
            mlflow.log_dict(model_report, "model_report.json")
            fig = self._candidates_figure(model_report, best_model_name)
            mlflow.log_figure(fig, "candidates_comparison.png")
            plt.close(fig)


    # ---------------------------------------------------------- final_model/
    def _is_better_than_production(self, new_f1: float) -> bool:
        """True if there is no production model yet, or the new one is at least as good."""
        path = self.model_trainer_config.final_metrics_file_path
        if not os.path.exists(path):
            return True
        current_f1 = read_yaml_file(path)["test_f1_score"]
        logging.info(f"Production test f1: {current_f1} | new test f1: {new_f1}")
        return new_f1 >= current_f1

    # ---------------------------------------------------------------- training
    def train_model(self, x_train, y_train, x_test, y_test) -> ModelTrainerArtifact:
        try:
            models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
            }
            params = {
                "Decision Tree": {
                    "criterion": ["gini", "entropy", "log_loss"],
                },
                "Random Forest": {
                    "n_estimators": [8, 16, 32, 128, 256],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Logistic Regression": {},
                "AdaBoost": {
                    "learning_rate": [0.1, 0.01, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }

            ## 2. tune and evaluate every model
            model_report: dict = evaluate_models(
                X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test,
                models=models, param=params,
            )

            ## 3. best model = highest CV f1 (chosen without looking at test)
            best_model_name = max(model_report, key=lambda name: model_report[name]["cv_f1_mean"])
            best = model_report[best_model_name]
            best_model = models[best_model_name]
            logging.info(f"Best model: {best_model_name} -> {best}")

            ## 4. checks on the chosen model
            cfg = self.model_trainer_config
            gap = best["train_f1"] - best["test_f1"]

            if best["test_f1"] < cfg.expected_accuracy:
                raise Exception(
                    f"Underfitting: best model {best_model_name} has test f1 {best['test_f1']} "
                    f"< expected {cfg.expected_accuracy}"
                )
            if gap > cfg.overfitting_underfitting_threshold:
                raise Exception(
                    f"Overfitting: {best_model_name} train f1 {best['train_f1']} vs test f1 {best['test_f1']} "
                    f"(gap {gap:.4f} > {cfg.overfitting_underfitting_threshold})"
                )
            if best["cv_f1_std"] > cfg.cv_std_warning:
                logging.warning(
                    f"Unstable model: {best_model_name} CV f1 std {best['cv_f1_std']} > {cfg.cv_std_warning}"
                )
            logging.info(f"Checks passed: test f1 {best['test_f1']}, gap {gap:.4f}, CV std {best['cv_f1_std']}")

            ## 5. metrics on train and test (phishing = 1 is the positive class)
            y_train_pred = best_model.predict(x_train)
            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)

            y_test_pred = best_model.predict(x_test)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

            logging.info(f"Train metrics: {classification_train_metric}")
            logging.info(f"Test metrics: {classification_test_metric}")

            ## track in MLflow (one run per training)
            self.track_mlflow(
                best_model_name, best_model, model_report,
                classification_train_metric, classification_test_metric,
            )

            ## 6. wrap preprocessor + model and save (Artifacts history)
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            os.makedirs(os.path.dirname(cfg.trained_model_file_path), exist_ok=True)

            network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
            save_object(cfg.trained_model_file_path, obj=network_model)

            ## 7. promote to production (final_model/) only if better than the current one
            new_f1 = float(classification_test_metric.f1_score)
            if self._is_better_than_production(new_f1):
                save_object(cfg.final_preprocessor_file_path, preprocessor)
                save_object(cfg.final_model_file_path, best_model)
                write_yaml_file(cfg.final_metrics_file_path, {
                    "algorithm": best_model_name,
                    "test_f1_score": new_f1,
                    "trained_model_file_path": cfg.trained_model_file_path,
                }, replace=True)
                logging.info(f"New model promoted to production: {best_model_name} (test f1 {new_f1})")
            else:
                logging.info("New model is worse than production. final_model/ kept unchanged")

            ## Model Trainer Artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=cfg.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric,
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise AppException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            ## 1. load arrays and split X / y (last column = target)
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr  = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test,  y_test  = test_arr[:, :-1],  test_arr[:, -1]

            return self.train_model(x_train, y_train, x_test, y_test)

        except Exception as e:
            raise AppException(e, sys)