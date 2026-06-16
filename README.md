# Part 4 — FastAPI Churn Scoring Service

## Goal
Expose the churn model through a simple internal API.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Train/save the API model

```bash
python src/train_model_for_api.py
```

This creates:

- `app/model.pkl`
- `sample_payload.json`

## Run the API

```bash
uvicorn app.main:app --reload
```

Open: `http://127.0.0.1:8000/docs`

## Sample requests

Health:

```bash
curl http://127.0.0.1:8000/health
```

Predict one customer:

```bash
curl -X POST http://127.0.0.1:8000/predict   -H "Content-Type: application/json"   -d @sample_payload.json
```

Batch predict:

```bash
curl -X POST http://127.0.0.1:8000/batch_predict   -H "Content-Type: application/json"   -d "[$(cat sample_payload.json), $(cat sample_payload.json)]"
```

## Tests

```bash
pytest -q
```

## Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## Required endpoints

- `GET /health`
- `POST /predict`
- `POST /batch_predict`

## Developer Information

- **Developer:** Shashwat Singh
- **Student Code:** IITP_AIML_2506887
- **Email:** shashwatanshul@gmail.com

