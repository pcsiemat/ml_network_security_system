# Network Security: Phishing Detection (End-to-End ML)

End-to-end machine learning project that detects phishing websites. Data is stored in MongoDB Atlas and processed through a modular training pipeline.

## Pipeline

```text
MongoDB → Data Ingestion → Data Validation → Data Transformation → Model Trainer → Model Evaluation → Model Pusher
```

Each stage follows the same pattern: **Config** (input) → **Component** (does the work) → **Artifact** (output for the next stage).

| Stage | What it does | Status |
|---|---|---|
| ETL (CSV → MongoDB) | Loads the raw CSV into MongoDB Atlas | ✅ Done |
| Data Ingestion | Reads MongoDB, saves a raw copy, stratified train/test split | ✅ Done |
| Data Validation | Checks column count and names against the schema, detects data drift (KS test) | ✅ Done |
| Data Transformation | Fills missing values (KNNImputer), maps target (phishing = 1, legitimate = 0), saves .npy arrays and the preprocessor | ✅ Done |
| Model Trainer | | ⏳ Next |
| Model Evaluation | | ⬜ |
| Model Pusher | | ⬜ |

## Project Structure

```text
data_schema/
└── schema.yaml        # Expected columns and types
ml_network_security_system/
├── components/        # Pipeline stages (data_ingestion.py, data_validation.py, ...)
├── constant/          # Fixed values (folder names, ratio, collection)
├── entity/            # config_entity.py (inputs), artifact_entity.py (outputs)
├── exception/         # Custom exception
├── logger/            # Logging setup
└── utils/             # Helpers (read/write YAML, ...)
push_data.py           # Loads the CSV into MongoDB
main.py                # Runs the pipeline
```

Output of each run:

```text
Artifacts/<timestamp>/
├── data_ingestion/
│   ├── feature_store/phisingData.csv
│   └── ingested/
│       ├── train.csv
│       └── test.csv
├── data_validation/
│   ├── validated/
│   │   ├── train.csv
│   │   └── test.csv
│   └── drift_report/
│       └── report.yaml
└── data_transformation/
    ├── transformed/
    │   ├── train.npy
    │   └── test.npy
    └── transformed_object/
        └── preprocessing.pkl
final_model/
└── preprocessor.pkl   # latest preprocessor, used for predictions
```

## Dataset

[UCI Phishing Websites](https://archive.ics.uci.edu/dataset/379/phishing+websites): 11,055 websites, 30 features, target `Result`.

| `Result` in the CSV | Meaning | In the model |
|---|---|---|
| -1 | phishing (4,898) | **1** |
| 1 | legitimate (6,157) | **0** |

Phishing is mapped to 1 so that precision, recall and f1 measure how well the model catches attacks.

The CSV is not in the repo (`*.csv` is in `.gitignore`). Download it and save it as:

```text
Network_Data/phisingData.csv
```

## Requirements

- [uv](https://docs.astral.sh/uv/) and Python 3.12
- A free [MongoDB Atlas](https://www.mongodb.com/cloud/atlas/register) cluster:
  1. Create a cluster.
  2. **Database Access** → create a database user (username + password).
  3. **Network Access** → add your IP address.
  4. **Connect → Drivers → Python** → copy the connection string.

## Setup

```bash
git clone https://github.com/<your_user>/ml-end-network-security-system.git
cd ml-end-network-security-system
uv sync
cp .env.example .env
```

Open `.env` and paste your connection string:

```text
MONGO_DB_URL="mongodb+srv://<user>:<password>@<cluster>.<id>.mongodb.net/"
```

Database and collection names are set in `ml_network_security_system/constant/training_pipeline/__init__.py` (`Cmatos_DB` / `Network_Data_Collection`). They are created automatically on the first insert.

## Run

```bash
uv run test_mongodb.py   # optional: check the connection
uv run push_data.py      # one time: load the CSV into MongoDB
uv run main.py           # run the pipeline
```

Each run creates a new `Artifacts/<timestamp>/` folder and overwrites `final_model/preprocessor.pkl`.
