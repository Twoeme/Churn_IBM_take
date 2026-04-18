# Product Name

ChurnGuard: Customer Retention Intelligence System

# Objective

Predict which customers are likely to churn within the next 30–90 days and estimate the financial impact of intervention strategies.

# Problem Statement

Subscription businesses lose revenue due to undetected churn risk. Current reporting is reactive (after churn happens), not predictive.

Success Metrics

Model metrics

AUC-ROC ≥ 0.80
Precision@Top10% ≥ 0.60
Recall ≥ 0.70 (for churn class)

Business metrics

% of churners correctly identified in top-risk segment
Estimated retained revenue:
Retained Revenue = (# prevented churn) × (Avg MRR)
Lift vs random targeting
Users
Growth / Marketing teams
Customer Success Managers
Founders / operators (like your use case in Tecvolución)
# Core Features
1. Churn Prediction Model
- Binary classifier:
- XGBoost
- Random Forest
- Output: probability of churn per user
2. Imbalance Handling
- SMOTE
- Compare with:
-   Class weights
-   Undersampling
3. Feature Engineering Layer
- Behavioral:
- Recency (last activity)
- Frequency (usage count)
- Monetary (billing)
- Subscription:
-   Contract type
-   Tenure
- Engagement:
-   Logins / actions per week
4. Risk Segmentation
- High risk (top 10%)
- Medium risk
- Low risk
5. Business Translation Layer
- Convert model output into money:
- Expected churn revenue loss
- Expected savings if intervention applied


# Data Pipeline

Input

Raw dataset (CSV / SQL / EXCEL)

Processing

Missing value imputation
Encoding (One-hot / Target encoding)
Scaling (if needed)

Output

Clean training dataset
Feature importance report


# Model Pipeline
Train/test split (time-aware if possible)
Baseline: Logistic Regression
Advanced:
XGBoost
Random Forest
Hyperparameter tuning
Cross-validation
Evaluation Strategy

Do NOT just show accuracy.

Focus on:

AUC-ROC
Precision-Recall curve
Confusion matrix
Lift curve (important for business)
Business Impact Simulation

Example:

Total users: 10,000
Churn rate: 20% → 2,000 users
Avg MRR: $20

If model captures:

70% of churners in top 20% segment → 1,400 users

If intervention success = 30%:

Saved users = 420

Revenue saved:

420 × $20 = $8,400/month

This is what makes the project stand out.

Deliverables
Jupyter notebook (clean + narrative)
Dashboard (Power BI or Streamlit)
README explaining:
Problem
Approach
Business value
Optional Advanced Layer (this is where you differentiate)
SHAP values for explainability
Time-series churn prediction
Uplift modeling (who to target, not just who will churn)
Deployment:
API with FastAPI
Batch scoring pipeline
Suggested Stack
Python (Pandas, Scikit-learn, XGBoost)
Visualization: Power BI (you already use it)
Optional: Streamlit for demo