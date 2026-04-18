import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from churnguard.pipeline import (
    business_impact,
    clean_data,
    engineer_features,
    encode_and_scale,
    evaluate_one,
    load_data,
    segment_risk,
    train_all,
)


st.set_page_config(page_title="ChurnGuard (IBM Telco)", layout="wide")


@st.cache_data(show_spinner=False)
def get_prepared_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    raw = load_data()
    clean = clean_data(raw)
    feat = engineer_features(clean)
    X, y, _ = encode_and_scale(feat)
    return feat, X, y


@st.cache_resource(show_spinner=False)
def train_cached(X: pd.DataFrame, y: pd.Series, imbalance_strategy: str, seed: int):
    return train_all(X, y, imbalance_strategy=imbalance_strategy, random_state=seed)


def fmt_money(x: float) -> str:
    return f"${x:,.2f}"


st.title("ChurnGuard: Customer Retention Intelligence System")
st.caption(
    "Predict churn risk and translate it into expected retained revenue (SMOTE vs class weights; "
    "LogReg / Random Forest / XGBoost; AUC/PR/Lift/Precision@Top10%)."
)

with st.sidebar:
    st.header("Controls")
    imbalance_strategy = st.radio(
        "Imbalance handling",
        options=[("SMOTE", "smote"), ("Class weights", "class_weight")],
        format_func=lambda x: x[0],
        index=0,
    )[1]
    seed = st.number_input("Random seed", min_value=1, max_value=10_000, value=42, step=1)
    avg_mrr = st.number_input("Avg MRR ($)", min_value=0.0, value=20.0, step=1.0)
    intervention_rate = st.slider("Intervention success rate", 0.0, 1.0, 0.30, 0.05)

feat_df, X, y = get_prepared_data()

col_a, col_b, col_c = st.columns(3)
col_a.metric("Customers", f"{len(X):,}")
col_b.metric("Churn rate", f"{(y.mean() * 100):.1f}%")
col_c.metric("Features", f"{X.shape[1]:,}")

with st.spinner("Training models…"):
    models, split = train_cached(X, y, imbalance_strategy, int(seed))

eval_rows = []
eval_details: dict[str, dict] = {}
for name, model in models.items():
    d = evaluate_one(model, split.X_test, split.y_test)
    eval_details[name] = d
    eval_rows.append(
        {
            "model": name,
            "auc_roc": d["auc_roc"],
            "pr_auc": d["pr_auc"],
            "recall_churn": d["recall_churn"],
            "precision@top10%": d["precision_top10pct"],
        }
    )

leaderboard = pd.DataFrame(eval_rows).sort_values("auc_roc", ascending=False)
best_model_name = leaderboard.iloc[0]["model"]
best_model = models[best_model_name]

st.subheader("Model leaderboard")
st.dataframe(
    leaderboard.style.format(
        {
            "auc_roc": "{:.3f}",
            "pr_auc": "{:.3f}",
            "recall_churn": "{:.3f}",
            "precision@top10%": "{:.3f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)
st.info(f"Best model by AUC-ROC: **{best_model_name}**", icon="ℹ️")

tab_eval, tab_risk, tab_business = st.tabs(["Evaluation", "Risk Segmentation", "Business Impact"])

with tab_eval:
    model_choice = st.selectbox("Pick a model to inspect", leaderboard["model"].tolist(), index=0)
    d = eval_details[model_choice]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC-ROC", f"{d['auc_roc']:.3f}")
    c2.metric("PR-AUC", f"{d['pr_auc']:.3f}")
    c3.metric("Recall (churn)", f"{d['recall_churn']:.3f}")
    c4.metric("Precision@Top10%", f"{d['precision_top10pct']:.3f}")

    p, r = d["pr_curve"]
    fig_pr = px.line(
        x=r,
        y=p,
        labels={"x": "Recall", "y": "Precision"},
        title="Precision–Recall curve",
    )
    fig_pr.update_layout(height=380, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig_pr, use_container_width=True)

    cm = d["confusion_matrix"]
    fig_cm = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Pred 0", "Pred 1"],
            y=["True 0", "True 1"],
            text=cm,
            texttemplate="%{text}",
            colorscale="Blues",
        )
    )
    fig_cm.update_layout(title="Confusion matrix", height=380, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig_cm, use_container_width=True)

    lift_df = d["lift"]
    fig_lift = px.bar(
        lift_df,
        x="cumulative_share",
        y="lift",
        title="Lift curve (deciles, higher is better)",
        labels={"cumulative_share": "Top share of customers targeted", "lift": "Lift vs baseline"},
    )
    fig_lift.update_layout(height=380, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig_lift, use_container_width=True)

with tab_risk:
    seg = segment_risk(X, best_model, high_q=0.90, medium_q=0.50)
    dist = seg["risk_segment"].value_counts().reindex(["High Risk", "Medium Risk", "Low Risk"])
    dist = dist.fillna(0).astype(int).reset_index()
    dist.columns = ["risk_segment", "count"]

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Risk distribution")
        st.dataframe(dist, use_container_width=True, hide_index=True)
    with c2:
        fig = px.pie(dist, names="risk_segment", values="count", hole=0.45, title="Risk tiers")
        fig.update_layout(height=380, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top risky customers (preview)")
    top = seg.sort_values("churn_probability", ascending=False).head(25).copy()
    top["churn_probability"] = top["churn_probability"].map(lambda v: round(float(v), 4))
    st.dataframe(top, use_container_width=True)

with tab_business:
    seg = segment_risk(X, best_model, high_q=0.90, medium_q=0.50)
    impact = business_impact(seg, avg_mrr=float(avg_mrr), intervention_rate=float(intervention_rate))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("High-risk customers", f"{impact['high_risk_count']:,}")
    c2.metric("Saved users (est.)", f"{impact['saved_users']:,}")
    c3.metric("Revenue saved / month", fmt_money(impact["revenue_saved_monthly"]))
    c4.metric("Revenue saved / year", fmt_money(impact["revenue_saved_annual"]))

    st.caption(
        "Formula: saved_users = high_risk_count × intervention_rate; revenue_saved = saved_users × avg_mrr."
    )
