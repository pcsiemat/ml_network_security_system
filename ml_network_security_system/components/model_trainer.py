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
            + track in MLflow: winner (params, metrics, registered model)
              and every candidate (metrics only, nested runs)
         6. wrap preprocessor + model in NetworkModel and save it as model.pkl
OUTPUT : ModelTrainerArtifact
=====================================================================
"""

import os
import sys
from dataclasses import asdict

import mlflow
import skops.io as sio
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
from ml_network_security_system.utils.main_utils import (
    evaluate_models,
    load_numpy_array_data,
    load_object,
    save_object,
)
from ml_network_security_system.utils.ml_utils.metric.classification_metric import get_classification_score
from ml_network_security_system.utils.ml_utils.model.estimator import NetworkModel
from ml_network_security_system.utils.main_utils.mlflow_utils import setup_mlflow


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

            ## configure MLflow once (local server, env var URI, or offline fallback)
            setup_mlflow()

        except Exception as e:
            raise AppException(e, sys)

    def track_mlflow(self, best_model_name, best_model, model_report, train_metric, test_metric):
        """
        One MLflow run per training:
        - parent run : winner params + train/test metrics + saved model (registered)
        - nested runs: metrics of every candidate, for comparison in the UI
        """
        # Types that skops flags as "untrusted" for THIS specific model.
        # Safe to trust them because we just trained this model ourselves.
        trusted_types = sio.get_untrusted_types(data=sio.dumps(best_model))

        with mlflow.start_run(run_name=f"training - {best_model_name}"):
            ## winner
            mlflow.set_tag("best_model", best_model_name)
            mlflow.log_params(best_model.get_params())

            # train_f1_score, test_f1_score, train_precision_score, ...
            for prefix, metric in (("train", train_metric), ("test", test_metric)):
                mlflow.log_metrics({f"{prefix}_{k}": v for k, v in asdict(metric).items()})

            mlflow.sklearn.log_model(
                best_model,
                name="model",
                skops_trusted_types=trusted_types,
                registered_model_name=best_model_name.replace(" ", "_"),
            )

            ## every candidate: metrics only (cv_f1_mean, cv_f1_std, train_f1, test_f1)
            for name, report in model_report.items():
                with mlflow.start_run(run_name=name, nested=True):
                    mlflow.set_tag("is_best", name == best_model_name)
                    mlflow.log_metrics(
                        {k: v for k, v in report.items() if isinstance(v, (int, float))}
                    )

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

            ## 6. wrap preprocessor + model and save (used by the app, independent of MLflow)
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            os.makedirs(os.path.dirname(cfg.trained_model_file_path), exist_ok=True)

            network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
            save_object(cfg.trained_model_file_path, obj=network_model)

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