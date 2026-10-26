FROM python:3.12-slim
WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir uv awscli
RUN uv sync --frozen --no-dev

CMD ["sh", "-c", "aws s3 sync s3://${TRAINING_BUCKET_NAME}/final_model final_model && uv run --no-sync uvicorn app:app --host 0.0.0.0 --port 8000"]