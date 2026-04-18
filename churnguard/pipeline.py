from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "Telco_customer_churn.xlsx"
TARGET_COL = "churn_value"


@dataclass(frozen=True)
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_excel(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[\s\-]+", "_", regex=True)
        .str.replace(r"[^\w]", "", regex=True)
    )

    cols_to_drop = [
        "customerid",
        "count",
        "country",
        "state",
        "lat_long",
        "latitude",
        "longitude",
        "churn_label",
        "churn_reason",
        "churn_score",
        "cltv",
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    if "total_charges" in df.columns:
        df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce").fillna(0)

    df = df.dropna().drop_duplicates()
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "tenure_months" in df.columns:
        max_tenure = float(df["tenure_months"].max()) if df["tenure_months"].max() else 1.0
        df["recency_score"] = (max_tenure - df["tenure_months"]) / max_tenure

    if all(c in df.columns for c in ["total_charges", "tenure_months"]):
        df["charges_per_month"] = df["total_charges"] / df["tenure_months"].clip(lower=1)

    if all(c in df.columns for c in ["monthly_charges", "total_charges"]):
        df["monthly_to_total_ratio"] = df["monthly_charges"] / df["total_charges"].clip(lower=1)

    addon_cols = [
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
    ]
    existing_addons = [c for c in addon_cols if c in df.columns]
    if existing_addons:
        addon_df = df[existing_addons].replace(
            {"Yes": 1, "No": 0, "No internet service": 0}
        )
        df["addon_service_count"] = addon_df.sum(axis=1)

    return df


def encode_and_scale(
    df: pd.DataFrame, target: str = TARGET_COL
) -> tuple[pd.DataFrame, pd.Series, StandardScaler]:
    df = df.copy()

    binary_cols = [
        col
        for col in df.columns
        if col != target and df[col].dropna().isin(["Yes", "No"]).all()
    ]
    if binary_cols:
        df[binary_cols] = df[binary_cols].replace({"Yes": 1, "No": 0})

    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    cat_cols = [c for c in df.select_dtypes(include="object").columns if c != target]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    y = df[target].astype(int)
    X = df.drop(columns=[target])

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    return X_scaled, y, scaler


def make_models(
    class_weight: str | None,
    y_train: pd.Series,
    random_state: int,
) -> dict[str, object]:
    # For XGBoost without SMOTE, scale_pos_weight can help. With SMOTE, keep it 1.
    scale_pos_weight = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
    if class_weight is None:
        scale_pos_weight = 1.0

    return {
        "Logistic Regression (baseline)": LogisticRegression(
            max_iter=2000, class_weight=class_weight, random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=random_state,
        ),
    }


def train_all(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    imbalance_strategy: str = "smote",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[dict[str, object], SplitData]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    if imbalance_strategy == "smote":
        smote = SMOTE(random_state=random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        class_weight = None
    elif imbalance_strategy == "class_weight":
        X_train_res, y_train_res = X_train, y_train
        class_weight = "balanced"
    else:
        raise ValueError("imbalance_strategy must be 'smote' or 'class_weight'")

    models = make_models(class_weight=class_weight, y_train=y_train_res, random_state=random_state)
    for model in models.values():
        model.fit(X_train_res, y_train_res)

    split = SplitData(X_train=X_train_res, X_test=X_test, y_train=y_train_res, y_test=y_test)
    return models, split


def precision_at_top_k(y_true: pd.Series, y_proba: np.ndarray, k_frac: float) -> float:
    k = max(1, int(len(y_true) * k_frac))
    top_idx = np.argsort(y_proba)[::-1][:k]
    return float(y_true.iloc[top_idx].mean())


def lift_curve(
    y_true: pd.Series, y_proba: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_true.values, "p": y_proba})
    df = df.sort_values("p", ascending=False).reset_index(drop=True)
    df["bin"] = pd.qcut(df.index + 1, q=n_bins, labels=False)

    baseline = df["y"].mean()
    agg = (
        df.groupby("bin", as_index=False)
        .agg(bin_size=("y", "size"), churn_rate=("y", "mean"))
        .sort_values("bin")
    )
    agg["cumulative_bin"] = agg["bin"] + 1
    agg["lift"] = agg["churn_rate"] / baseline if baseline > 0 else np.nan
    agg["cumulative_share"] = agg["cumulative_bin"] / n_bins
    return agg


def evaluate_one(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc_roc = float(roc_auc_score(y_test, y_proba))

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = float(auc(recall, precision))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    recall_churn = float(tp / max(1, (tp + fn)))

    p10 = precision_at_top_k(y_test, y_proba, 0.10)

    return {
        "auc_roc": auc_roc,
        "pr_auc": pr_auc,
        "recall_churn": recall_churn,
        "precision_top10pct": p10,
        "confusion_matrix": cm,
        "pr_curve": (precision, recall),
        "lift": lift_curve(y_test, y_proba, n_bins=10),
        "y_proba": y_proba,
    }


def segment_risk(
    X: pd.DataFrame, model, *, high_q: float = 0.90, medium_q: float = 0.50
) -> pd.DataFrame:
    proba = model.predict_proba(X)[:, 1]
    q_high = float(np.quantile(proba, high_q))
    q_med = float(np.quantile(proba, medium_q))

    segments = pd.cut(
        proba,
        bins=[-np.inf, q_med, q_high, np.inf],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )

    return pd.DataFrame(
        {"churn_probability": proba, "risk_segment": segments},
        index=X.index,
    )


def business_impact(
    segmented_df: pd.DataFrame,
    *,
    avg_mrr: float,
    intervention_rate: float,
) -> dict[str, float]:
    n_total = int(len(segmented_df))
    n_high = int((segmented_df["risk_segment"] == "High Risk").sum())
    saved = int(round(n_high * float(intervention_rate)))
    monthly = float(saved * avg_mrr)
    annual = float(monthly * 12)
    return {
        "total_customers": n_total,
        "high_risk_count": n_high,
        "saved_users": saved,
        "revenue_saved_monthly": monthly,
        "revenue_saved_annual": annual,
    }

