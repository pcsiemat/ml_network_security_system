import os
import sys

import certifi
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()
mongo_db_url = os.getenv("MONGODB_URL_KEY")
print(mongo_db_url)


import pymongo
from ml_network_security_system.exception.exception import AppException
from ml_network_security_system.logger.log import logging
from ml_network_security_system.pipeline.training_pipeline import TrainingPipeline

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd
from datetime import datetime

from ml_network_security_system.utils.main_utils.utils import load_object
from ml_network_security_system.utils.ml_utils.model.estimator import NetworkModel

client = pymongo.MongoClient(mongo_db_url, tlsCAFile = ca)

from ml_network_security_system.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME
from ml_network_security_system.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME

from ml_network_security_system.constant.training_pipeline import (
    FINAL_MODEL_OBJECT_DIR,
    MODEL_FILE_NAME,
    PREPROCESSING_OBJECT_FILE_NAME,
    TEMPLATE_DIR,
    PREDICTION_OUTPUT_DIR
)
FINAL_MODEL_PATH = os.path.join(FINAL_MODEL_OBJECT_DIR, MODEL_FILE_NAME)
FINAL_PREPROCESSOR_PATH = os.path.join(FINAL_MODEL_OBJECT_DIR, PREPROCESSING_OBJECT_FILE_NAME)


app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory=TEMPLATE_DIR)   # folder where table.html lives


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")
@app.get("/train")
async def train_route():
    try:
        trainingPipeline = TrainingPipeline()
        trainingPipeline.run_pipeline()
        return Response("Training is sucessful")
    except Exception as e:
        raise AppException(e,sys)


def make_predictions(file: UploadFile) -> pd.DataFrame:
    """Shared by both routes: read CSV, predict, label."""
    df = pd.read_csv(file.file)
    preprocessor = load_object(FINAL_PREPROCESSOR_PATH)
    model = load_object(FINAL_MODEL_PATH)
    network_model = NetworkModel(preprocessor=preprocessor, model=model)
    df["predicted_column"] = network_model.predict(df)
    df["predicted_column"] = df["predicted_column"].map({1: "Phishing", 0: "Legitimate"})

    ## keep a record of every prediction (one file per request, never overwritten)
    os.makedirs(PREDICTION_OUTPUT_DIR, exist_ok=True)
    filename = f"prediction_{datetime.now():%Y_%m_%d_%H_%M_%S}.csv"
    df.to_csv(os.path.join(PREDICTION_OUTPUT_DIR, filename), index=False)
    return df

@app.post("/predict")                       # HTML page 
async def predict_route(request: Request, file: UploadFile = File(...)):
    df = make_predictions(file)
    summary = {
        "total": len(df),
        "phishing": int((df["predicted_column"] == "Phishing").sum()),
        "legitimate": int((df["predicted_column"] == "Legitimate").sum()),
    }
    return templates.TemplateResponse(request, "table.html", {
        "columns": df.columns.tolist(),
        "rows": df.to_dict("records"),
        "summary": summary,
    })



if __name__ == "__main__":
    app_run(app, host = "localhost", port=8000)