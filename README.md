# Phishing Website Detection — End-to-End MLOps

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-package%20manager-DE5FE9)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white)
[![DagsHub](https://img.shields.io/badge/DagsHub-experiments-6A4CFF)](https://dagshub.com/pcsiemat/ml_network_security_system.mlflow/#/experiments/0)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20ECR%20%7C%20EC2-FF9900?logo=amazonwebservices&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)

A machine learning system that classifies websites as **phishing** or **legitimate**, built and deployed end to end: from raw data in MongoDB to a live API on AWS, updated automatically on every push.

![Architecture](docs/architecture.png)

## What it does

1. **Trains** a classifier on 11,055 websites (30 URL and page features) stored in MongoDB Atlas.
2. **Tracks** every experiment in MLflow / DagsHub and registers the best model.
3. **Serves** predictions through a FastAPI app: upload a CSV, get each site labeled as phishing or legitimate.
4. **Deploys** automatically: GitHub Actions builds a Docker image, pushes it to Amazon ECR and runs it on Amazon EC2.

## Highlights

- **Modular pipeline**: Data Ingestion → Data Validation (schema + drift test) → Data Transformation → Model Trainer.
- **Model selection**: 5 classifiers tuned with 5-fold GridSearchCV; best model chosen by cross-validated F1.
- **Quality gates**: training stops if the model underfits or overfits.
- **Safe promotion**: the production model is replaced only when the new one performs at least as well.
- **Experiment tracking**: nested MLflow runs, comparison chart and Model Registry with a `champion` alias.
- **Target re-encoded**: phishing = `1`, legitimate = `0`, so the metrics measure attacks caught.
- **Managed with uv**: Python 3.12, `pyproject.toml` + `uv.lock` for reproducible installs, also inside Docker.
- **CI/CD**: every push builds, tags (`commit` + `latest`) and deploys a new image; old images are cleaned up.

## Dataset and target

[UCI Phishing Websites](https://archive.ics.uci.edu/dataset/379/phishing+websites): 11,055 websites, 30 features coded `-1 / 0 / 1`.
The original target `Result` was re-encoded so that the positive class is the one we want to detect:

| Original `Result` | Meaning | Model target |
|---|---|---|
| `-1` | Phishing (4,898) | **`1`** |
| `1` | Legitimate (6,157) | **`0`** |

## Key design decisions

| Decision | Why |
|---|---|
| Phishing mapped to class `1` | Precision, recall and F1 measure attacks caught |
| Model chosen by cross-validated F1 on train | The test set stays untouched for the final check |
| Quality gates (underfitting, overfitting, unstable CV) | A bad model never reaches production |
| Production model promoted only if as good or better | A worse retrain cannot replace a good model |
| Parent run + nested candidate runs, Model Registry with `champion` alias | Every candidate is comparable and the production model is traceable |
| Model downloaded from S3 when the container starts | A new model needs a restart, not a new image |
| Images tagged with commit + `latest`, old images cleaned up | Easy rollback, low storage cost |
| uv with lock file, Python 3.12 slim image | Reproducible and fast builds |

## Results

| Metric | Value |
|---|---|
| Best model | Random Forest |
| Test F1 (phishing class) | 0.97 |
| Dataset | 11,055 websites · 44% phishing / 56% legitimate |

## Experiment tracking

Each training creates a parent run with 5 nested candidate runs (one per algorithm) and registers the best model.
Runs with source `uvicorn` were trained in the cloud through the API on EC2.

![MLflow runs on DagsHub](docs/experiments.png)

👉 [See all experiments on DagsHub](https://dagshub.com/pcsiemat/ml_network_security_system.mlflow/#/experiments/0)

## Tech stack

| Area | Tools |
|---|---|
| Language | Python 3.12, uv |
| Data | MongoDB Atlas, pandas, NumPy |
| Machine learning | scikit-learn, SciPy |
| Experiment tracking | MLflow, DagsHub |
| API | FastAPI, Jinja2 |
| Cloud and DevOps | Docker, GitHub Actions, AWS S3, ECR, EC2 |

## Run locally

The project uses [uv](https://docs.astral.sh/uv/) instead of pip and `requirements.txt`: `uv sync` creates the virtual environment and installs the exact versions from `uv.lock`.

```bash
git clone https://github.com/pcsiemat/ml_network_security_system.git
cd ml_network_security_system
uv sync
cp .env.sample .env              # add your MongoDB connection string
uv run scripts/push_data.py      # load the dataset (one time)
uv run main.py                   # train
uv run app.py                    # open http://localhost:8000/docs
```

Try `POST /predict` with the sample file `valid_data/test_phishing.csv`.

## Project structure

```text
ml_network_security_system/   pipeline code: components, config, entities, utils
.github/workflows/main.yml    CI/CD workflow
app.py                        FastAPI app (/train, /predict)
main.py                       runs the training pipeline
Dockerfile                    API image
docs/images/                  README images
```

## Next steps

- Model evaluation stage comparing champion and challenger on the same test set.
- Unit tests and linting in the CI job.
- IAM roles and AWS Secrets Manager instead of access keys.

## Author

**Edilberto Matos** — [LinkedIn](https://www.linkedin.com/in/christian-matos-sanchez) · [GitHub](https://github.com/pcsiemat)
