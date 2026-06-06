# Monitoring Plan and Responsible Use

## What to monitor after deployment

1. **Data drift:** Track distributions of `recency_days`, `frequency_180d`, `monetary_180d`, support features, return rate, and web/app activity. Alert when they move materially from the training snapshot.
2. **Prediction distribution:** Track the share of low/medium/high risk customers and the average churn probability by week. Sudden changes can indicate upstream data bugs or business changes.
3. **Business outcomes:** Track retention conversion, repeat purchase rate, revenue saved, campaign cost, unsubscribe rate, and customer complaints by segment and intervention type.
4. **API health:** Track latency, error rate, validation failures, request volume, and model-loading failures.
5. **Calibration and accuracy:** When actual 60-day outcomes become available, compare predicted probabilities against observed churn and review false positives/false negatives.
6. **Retraining triggers:** Retrain if PR-AUC/recall drops, calibration worsens, new campaigns/products launch, data definitions change, or customer behavior shifts seasonally.

## Responsible use

The API should support retention prioritization, not replace human judgment. High-risk predictions should be reviewed alongside customer context, support history, and campaign eligibility. Do not use churn scores to deny service, manipulate customers, or exclude groups unfairly. Expensive discounts should be tested with holdout groups so the team measures true incremental retention lift.
