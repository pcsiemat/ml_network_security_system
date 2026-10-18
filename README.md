# Network Security: Phishing Detection (End-to-End ML)

End-to-end machine learning project that detects phishing websites. Data is stored in MongoDB Atlas and processed through a modular training pipeline.

## Pipeline

```text
MongoDB → Data Ingestion → Data Validation → Data Transformation → Model Trainer → Model Evaluation → Model Pusher
```

Each stage follows the same pattern: **Config** (input) → **Component** (does the work) → **Artifact** (output for the next stage).

| Stage | Status |
|---|---|
| ETL (CSV → MongoDB) | ✅ Done |
| Data Ingestion | ✅ Done |
| Data Validation | ⏳ Next |
| Data Transformation | ⬜ |
| Model Trainer | ⬜ |
| Model Evaluation | ⬜ |
| Model Pusher | ⬜ |

## Project Structure

```text
ml_network_security_system/
├── components/        # Pipeline stages (data_ingestion.py, ...)
├── constant/          # Fixed values (folder names, ratio, collection)
├── entity/            # config_entity.py (inputs), artifact_entity.py (outputs)
├── exception/         # Custom exception
└── logger/            # Logging setup
push_data.py           # Loads the CSV into MongoDB
main.py                # Runs the pipeline
```

Output of each run:

```text
Artifacts/<timestamp>/
└── data_ingestion/
    ├── feature_store/phisingData.csv
    └── ingested/
        ├── train.csv
        └── test.csv
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
git clone https://github.com/<your_user>/ml-end-network-security-system.git
cd ml-end-network-security-system
uv sync
```

Create a `.env` file in the project root:

```text
cp .env.example .env    # then put your real MONGO_DB_URL in .env

MONGO_DB_URL="mongodb+srv://<user>:<password>@<cluster>.mongodb.net/"
```

## Run

```bash
uv run push_data.py   # one time: load the CSV into MongoDB
uv run main.py        # run the pipeline
```


