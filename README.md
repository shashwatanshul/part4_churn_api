# Part 4 — FastAPI Churn Scoring Service

## Goal

Expose the churn model through a simple internal API that can be used by a CRM or retention team to score customer churn risk.

The API returns:

* churn probability
* predicted churn class
* risk band
* short risk explanation

## Project Structure

```text
part4_churn_api/
├── app/
│   ├── main.py              # FastAPI application (endpoints, logic, schemas)
│   └── model.pkl            # Trained model binary loaded by the API
├── data/                    # Raw capstone datasets & data dictionary
├── src/
│   └── train_model_for_api.py # Script to train and save the model for the API
├── tests/
│   └── test_api.py          # API unit tests (endpoints, validation checks)
├── Dockerfile               # Docker configuration to containerize the service
├── monitoring_plan.md       # Metrics, drift, and post-deployment monitoring guide
├── README.md                # Project documentation and developer info
├── requirements.txt         # Python dependencies
└── sample_payload.json      # Sample request payload for API testing
```

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

* `app/model.pkl`
* `sample_payload.json`

## Run the API

```bash
uvicorn app.main:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

You can test the `/health`, `/predict`, and `/batch_predict` endpoints from Swagger UI.

## Required Endpoints

* `GET /health`
* `POST /predict`
* `POST /batch_predict`

## Sample Requests Using cURL

### Health

```bash
curl http://127.0.0.1:8000/health
```

### Predict one customer

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_payload.json
```

### Batch predict

```bash
curl -X POST http://127.0.0.1:8000/batch_predict \
  -H "Content-Type: application/json" \
  -d "[$(cat sample_payload.json), $(cat sample_payload.json)]"
```

## API Usage

After starting the server:

```bash
uvicorn app.main:app --reload
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

You can test the endpoints from Swagger UI using the sample payloads below.

## `GET /health`

Sample response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "snapshot_date": "2025-09-30",
  "feature_count": 25
}
```

## `POST /predict`

Sample request:

```json
{
  "city_tier": "Tier 1",
  "age_group": "18-24",
  "acquisition_channel": "Instagram",
  "loyalty_tier": "Silver",
  "preferred_category": "Makeup",
  "marketing_consent": "Yes",
  "recency_days": 107,
  "frequency_180d": 1,
  "monetary_180d": 362.73,
  "return_rate_180d": 0.0,
  "avg_discount_pct_180d": 0.23,
  "avg_rating_180d": 3.0,
  "category_diversity_180d": 1,
  "ticket_count_90d": 0,
  "negative_ticket_rate_90d": 0.0,
  "avg_resolution_hours_90d": 0.0,
  "days_since_signup": 524,
  "sessions_30d": 1,
  "product_views_30d": 4,
  "cart_adds_30d": 0,
  "wishlist_adds_30d": 0,
  "abandoned_carts_30d": 0,
  "email_opens_30d": 2,
  "campaign_clicks_30d": 0,
  "last_visit_days_ago": 20,
  "customer_id": "CUST00001"
}
```

Sample response:

```json
{
  "customer_id": "CUST00001",
  "churn_probability": 0.82,
  "predicted_class": 1,
  "threshold": 0.45,
  "risk_band": "high",
  "risk_explanation": "Last purchase is stale; recent app/web engagement is low. Model probability=0.820."
}
```

The exact probability may differ depending on the trained model artifact.

## `POST /batch_predict`

Sample request:

```json
[
  {
    "city_tier": "Tier 1",
    "age_group": "18-24",
    "acquisition_channel": "Instagram",
    "loyalty_tier": "Silver",
    "preferred_category": "Makeup",
    "marketing_consent": "Yes",
    "recency_days": 107,
    "frequency_180d": 1,
    "monetary_180d": 362.73,
    "return_rate_180d": 0.0,
    "avg_discount_pct_180d": 0.23,
    "avg_rating_180d": 3.0,
    "category_diversity_180d": 1,
    "ticket_count_90d": 0,
    "negative_ticket_rate_90d": 0.0,
    "avg_resolution_hours_90d": 0.0,
    "days_since_signup": 524,
    "sessions_30d": 1,
    "product_views_30d": 4,
    "cart_adds_30d": 0,
    "wishlist_adds_30d": 0,
    "abandoned_carts_30d": 0,
    "email_opens_30d": 2,
    "campaign_clicks_30d": 0,
    "last_visit_days_ago": 20,
    "customer_id": "CUST00001"
  }
]
```

Sample response:

```json
[
  {
    "customer_id": "CUST00001",
    "churn_probability": 0.82,
    "predicted_class": 1,
    "threshold": 0.45,
    "risk_band": "high",
    "risk_explanation": "Last purchase is stale; recent app/web engagement is low. Model probability=0.820."
  }
]
```

## Tests

Run the API tests from the project root:

```bash
pytest -q
```

Expected result:

```text
4 passed
```

## Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

After running the Docker container, open:

```text
http://127.0.0.1:8000/docs
```

## Model and Data Notes

* The model is trained using the provided `rfm_modeling_snapshot.csv`.
* The snapshot date is `2025-09-30`.
* The target variable is `churn_next_60d`, which represents whether the customer churned in the next 60 days.
* The target column, customer ID, snapshot date, and split column are not used as model input features.
* The API loads the saved model from `app/model.pkl`.
* If `app/model.pkl` is missing, run:

```bash
python src/train_model_for_api.py
```

## Validation Behaviour

The API uses Pydantic validation.

Invalid or incomplete request payloads return HTTP `422 Validation Error`. This is expected FastAPI behaviour and prevents invalid customer feature data from being scored.

## Responsible Use

The churn score should be used as a decision-support signal for retention planning. It should not be used as the only basis for customer treatment.

Recommended use:

* prioritize high-risk customers for review
* design targeted retention actions
* monitor churn-risk trends
* combine prediction output with business context

Not recommended:

* blindly giving discounts to every high-risk customer
* making irreversible customer decisions only from model output
* using the model without monitoring data drift and prediction quality

## Monitoring Notes

After deployment, the following should be monitored:

* input data drift
* prediction distribution
* percentage of high-risk customers
* API errors and latency
* retention campaign outcomes
* actual churn rate after intervention
* model retraining triggers

Retraining should be considered if customer behaviour changes significantly, prediction quality drops, or business campaigns change the churn pattern.

## Developer Information

* **Developer:** Shashwat Singh
* **Student Code:** IITP_AIML_2506887
* **Email:** shashwatanshul@gmail.com
