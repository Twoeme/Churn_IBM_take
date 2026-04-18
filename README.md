# ChurnGuard (IBM Telco) — Churn_IBM_take

Predict customer churn risk (30–90 days proxy) and translate it into estimated revenue impact, aligned with `RPD_Churn.md`.

## What you get

- **End-to-end churn pipeline**: load → clean → feature engineering → encode/scale → imbalance handling → train → evaluate
- **Models**: Logistic Regression (baseline), Random Forest, XGBoost
- **Imbalance strategies**: SMOTE vs class weights
- **Evaluation**: AUC-ROC, PR curve (AUC), confusion matrix, lift curve, Precision@Top10%
- **Risk tiers**: High (top 10%), Medium (50–90%), Low (bottom 50%)
- **Business impact**: revenue saved simulation based on intervention success rate and Avg MRR

## Run the dashboard (Streamlit)

From the repository root:

```bash
cd Churn_IBM_take
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Data

The app expects the Excel file:

- `Telco_customer_churn.xlsx`

If you rename/move it, update the path in `churnguard/pipeline.py`.

## Project structure

- `app.py`: Streamlit dashboard
- `churnguard/pipeline.py`: data + ML pipeline (training/evaluation/segmentation/business impact)

