from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


st.set_page_config(
    page_title="Adult Income Analytics",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


DATA_PATH = "adult_cleaned.xlsx"
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "age",
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "hours_per_week",
    "native_country",
    "capital_gain_flag",
    "capital_loss_flag",
]

NUMERIC_FEATURES = ["age", "hours_per_week"]
BINARY_FEATURES = ["capital_gain_flag", "capital_loss_flag"]
CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

MODEL_NOTES = {
    "Logistic Regression": "Linear baseline with strong interpretability. It often has high recall, but lower precision.",
    "Decision Tree": "Interpretable non-linear tree-based model. It captures simple decision rules but can overfit if not controlled.",
    "Random Forest": "Final selected model by F1-score. It balances non-linear patterns with stable test performance.",
    "XGBoost": "Boosting-style model with strong ranking ability. If xgboost is unavailable, the app uses HistGradientBoosting as a local fallback.",
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #17212b;
            --muted: #667085;
            --line: #d9e2ec;
            --blue: #2563eb;
            --green: #0f9f8f;
            --amber: #b7791f;
            --rose: #e11d48;
            --soft: #f7fafc;
        }
        html, body, [class*="css"] {
            font-size: 19px;
        }
        .main .block-container {
            padding-top: 1.3rem;
            padding-bottom: 2rem;
            max-width: 1280px;
        }
        h1, h2, h3 {
            color: var(--ink);
            letter-spacing: 0;
        }
        h1 {
            font-size: 2.35rem;
        }
        h2 {
            font-size: 1.75rem;
        }
        h3 {
            font-size: 1.35rem;
        }
        p, li, label, span, div, input, textarea, select {
            font-size: 1rem;
        }
        [data-testid="stSidebar"] {
            background: #17212b;
        }
        [data-testid="stSidebar"] [data-testid="stSidebarNav"] {
            display: none;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label {
            color: #d7dee8;
        }
        [data-testid="stSidebar"] .stButton > button {
            width: 100%;
            justify-content: flex-start;
            background: transparent;
            color: #d7dee8;
            border: 0;
            border-radius: 8px;
            padding: 0.75rem 0.9rem;
            font-size: 1rem;
            font-weight: 600;
            box-shadow: none;
            transition: background 150ms ease, color 150ms ease, transform 150ms ease;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            background: #ffffff;
            color: #17212b;
            border: 0;
            transform: translateX(2px);
        }
        [data-testid="stSidebar"] .stButton > button:focus {
            box-shadow: none;
            border: 0;
        }
        .nav-active {
            background: #ffffff;
            color: #17212b;
            border-radius: 8px;
            padding: 0.75rem 0.9rem;
            margin: 0 0 0.35rem 0;
            font-size: 1rem;
            font-weight: 700;
            box-shadow: 0 8px 22px rgba(0, 0, 0, 0.14);
        }
        div[data-testid="stDeployButton"] {
            display: none;
        }
        [data-testid="stStatusWidget"] {
            display: none;
        }
        .stSelectbox label,
        .stNumberInput label,
        .stSlider label,
        .stRadio label,
        .stMultiSelect label,
        .stFileUploader label {
            font-size: 1.02rem;
            font-weight: 650;
        }
        .stSelectbox div,
        .stNumberInput input,
        .stSlider div,
        .stRadio div,
        .stMultiSelect div,
        .stFileUploader div,
        .stDataFrame div {
            font-size: 1rem;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }
        .insight-card {
            background: #ffffff;
            border: 1px solid var(--line);
            border-left: 4px solid var(--green);
            border-radius: 8px;
            padding: 14px 16px;
            margin: 8px 0 12px 0;
            color: var(--ink);
        }
        .warning-card {
            background: #fff8ed;
            border: 1px solid #f4d29c;
            border-left: 4px solid var(--amber);
            border-radius: 8px;
            padding: 14px 16px;
            margin: 8px 0 12px 0;
            color: #533b12;
        }
        .prediction-card {
            background: linear-gradient(135deg, #f8fbff 0%, #eefbf8 100%);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 18px;
            margin: 8px 0 12px 0;
        }
        .small-muted {
            color: var(--muted);
            font-size: 0.92rem;
        }
        .stButton>button {
            border-radius: 8px;
            border: 1px solid #1d4ed8;
            background: #2563eb;
            color: #ffffff;
            font-weight: 600;
            font-size: 1rem;
            min-height: 2.7rem;
        }
        .stDownloadButton>button {
            border-radius: 8px;
            font-size: 1rem;
            min-height: 2.7rem;
        }
        .app-loader-wrap {
            min-height: 68vh;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            gap: 16px;
            color: #475467;
        }
        .app-loader {
            width: 48px;
            height: 48px;
            border: 5px solid #d9e2ec;
            border-top-color: #2563eb;
            border-right-color: #0f9f8f;
            border-radius: 50%;
            animation: app-spin 850ms linear infinite;
        }
        @keyframes app-spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def loading_ring(message: str) -> str:
    return f"""
    <div class="app-loader-wrap">
        <div class="app-loader"></div>
        <div>{message}</div>
    </div>
    """


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH)
    df = df.copy()
    for col in CATEGORICAL_FEATURES + ["income"]:
        df[col] = df[col].astype(str).str.strip().str.lower()
    return df


def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            ("cat", make_ohe(), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def make_models() -> dict[str, Pipeline]:
    xgb_model = (
        XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.2,
            subsample=1.0,
            colsample_bytree=1.0,
            min_child_weight=1,
            reg_lambda=5,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        if XGBOOST_AVAILABLE
        else HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.08,
            max_leaf_nodes=31,
            random_state=RANDOM_STATE,
        )
    )

    return {
        "Logistic Regression": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        C=1,
                        penalty="l2",
                        solver="liblinear",
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    DecisionTreeClassifier(
                        criterion="entropy",
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=20,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=None,
                        min_samples_split=10,
                        min_samples_leaf=1,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                ("model", xgb_model),
            ]
        ),
    }


@dataclass
class ModelBundle:
    models: dict[str, Pipeline]
    metrics: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    baseline_row: dict


@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame) -> ModelBundle:
    data = df.copy()
    y = (data["income"] == ">50k").astype(int)
    X = data[FEATURE_COLUMNS]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = make_models()
    metric_rows = []
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        metric_rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "f1": f1_score(y_test, pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, prob),
                "pr_auc": average_precision_score(y_test, prob),
            }
        )

    baseline_row = {}
    for col in FEATURE_COLUMNS:
        if col in NUMERIC_FEATURES:
            baseline_row[col] = float(X_train[col].median())
        elif col in BINARY_FEATURES:
            baseline_row[col] = int(X_train[col].mode().iloc[0])
        else:
            baseline_row[col] = str(X_train[col].mode().iloc[0])

    return ModelBundle(
        models=models,
        metrics=pd.DataFrame(metric_rows),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        baseline_row=baseline_row,
    )


def predict_probability(model: Pipeline, row: pd.DataFrame) -> float:
    return float(model.predict_proba(row[FEATURE_COLUMNS])[:, 1][0])


def class_from_probability(probability: float, threshold: float) -> str:
    return ">50K" if probability >= threshold else "<=50K"


def confidence_label(probability: float, threshold: float) -> tuple[str, float]:
    distance = abs(probability - threshold)
    if distance >= 0.25:
        label = "High"
    elif distance >= 0.12:
        label = "Medium"
    else:
        label = "Low"
    return label, distance


def get_options(df: pd.DataFrame, col: str) -> list[str]:
    return sorted(df[col].dropna().astype(str).unique().tolist())


def input_form(df: pd.DataFrame, key_prefix: str, defaults: dict | None = None) -> dict:
    defaults = defaults or {}
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input(
            "Age",
            min_value=int(df["age"].min()),
            max_value=int(df["age"].max()),
            value=int(defaults.get("age", 38)),
            step=1,
            key=f"{key_prefix}_age",
        )
        education = st.selectbox(
            "Education",
            get_options(df, "education"),
            index=get_default_index(df, "education", defaults.get("education", "bachelors")),
            key=f"{key_prefix}_education",
        )
        occupation = st.selectbox(
            "Occupation",
            get_options(df, "occupation"),
            index=get_default_index(df, "occupation", defaults.get("occupation", "prof-specialty")),
            key=f"{key_prefix}_occupation",
        )
        capital_gain_flag = st.selectbox(
            "Capital Gain Flag",
            [0, 1],
            index=int(defaults.get("capital_gain_flag", 0)),
            key=f"{key_prefix}_capital_gain_flag",
        )
    with c2:
        hours = st.number_input(
            "Hours per Week",
            min_value=int(df["hours_per_week"].min()),
            max_value=int(df["hours_per_week"].max()),
            value=int(defaults.get("hours_per_week", 40)),
            step=1,
            key=f"{key_prefix}_hours",
        )
        workclass = st.selectbox(
            "Workclass",
            get_options(df, "workclass"),
            index=get_default_index(df, "workclass", defaults.get("workclass", "private")),
            key=f"{key_prefix}_workclass",
        )
        marital = st.selectbox(
            "Marital Status",
            get_options(df, "marital_status"),
            index=get_default_index(df, "marital_status", defaults.get("marital_status", "married-civ-spouse")),
            key=f"{key_prefix}_marital",
        )
        capital_loss_flag = st.selectbox(
            "Capital Loss Flag",
            [0, 1],
            index=int(defaults.get("capital_loss_flag", 0)),
            key=f"{key_prefix}_capital_loss_flag",
        )
    with c3:
        sex = st.selectbox(
            "Sex",
            get_options(df, "sex"),
            index=get_default_index(df, "sex", defaults.get("sex", "male")),
            key=f"{key_prefix}_sex",
        )
        race = st.selectbox(
            "Race",
            get_options(df, "race"),
            index=get_default_index(df, "race", defaults.get("race", "white")),
            key=f"{key_prefix}_race",
        )
        relationship = st.selectbox(
            "Relationship",
            get_options(df, "relationship"),
            index=get_default_index(df, "relationship", defaults.get("relationship", "husband")),
            key=f"{key_prefix}_relationship",
        )
        country = st.selectbox(
            "Native Country",
            get_options(df, "native_country"),
            index=get_default_index(df, "native_country", defaults.get("native_country", "united-states")),
            key=f"{key_prefix}_country",
        )

    return {
        "age": int(age),
        "workclass": workclass,
        "education": education,
        "marital_status": marital,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "hours_per_week": int(hours),
        "native_country": country,
        "capital_gain_flag": int(capital_gain_flag),
        "capital_loss_flag": int(capital_loss_flag),
    }


def get_default_index(df: pd.DataFrame, col: str, value: str) -> int:
    options = get_options(df, col)
    return options.index(value) if value in options else 0


def row_to_frame(row: dict) -> pd.DataFrame:
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def local_explanation(
    model: Pipeline,
    row: dict,
    baseline_row: dict,
    top_n: int = 8,
) -> pd.DataFrame:
    original = row_to_frame(row)
    original_prob = predict_probability(model, original)
    rows = []
    for feature in FEATURE_COLUMNS:
        perturbed = row.copy()
        perturbed[feature] = baseline_row[feature]
        perturbed_prob = predict_probability(model, row_to_frame(perturbed))
        rows.append(
            {
                "feature": feature,
                "current_value": row[feature],
                "baseline_value": baseline_row[feature],
                "contribution": original_prob - perturbed_prob,
            }
        )
    return (
        pd.DataFrame(rows)
        .assign(abs_contribution=lambda d: d["contribution"].abs())
        .sort_values("abs_contribution", ascending=False)
        .head(top_n)
    )


def plot_local_explanation(explanation: pd.DataFrame) -> go.Figure:
    plot_df = explanation.sort_values("contribution")
    colors = ["#e11d48" if x < 0 else "#0f9f8f" for x in plot_df["contribution"]]
    fig = go.Figure(
        go.Bar(
            x=plot_df["contribution"],
            y=plot_df["feature"],
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>Contribution: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis_title="Change in P(>50K) versus baseline",
        yaxis_title=None,
        template="plotly_white",
    )
    fig.add_vline(x=0, line_color="#98a2b3", line_width=1)
    return fig


def prediction_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict:
    pred = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def metric_cards(metrics: dict | pd.Series) -> None:
    cols = st.columns(6)
    labels = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1-score", "f1"),
        ("ROC-AUC", "roc_auc"),
        ("PR-AUC", "pr_auc"),
    ]
    for col, (label, key) in zip(cols, labels):
        col.metric(label, f"{float(metrics[key]):.3f}")


def plot_confusion(y_true: pd.Series, probabilities: np.ndarray, threshold: float, title: str) -> go.Figure:
    pred = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred)
    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["<=50K", ">50K"],
        y=["<=50K", ">50K"],
        color_continuous_scale="Blues",
        title=title,
    )
    fig.update_layout(template="plotly_white", height=390, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def plot_roc_curves(bundle: ModelBundle, selected_models: list[str]) -> go.Figure:
    fig = go.Figure()
    for name in selected_models:
        probabilities = bundle.models[name].predict_proba(bundle.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(bundle.y_test, probabilities)
        auc = roc_auc_score(bundle.y_test, probabilities)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} AUC={auc:.3f}"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash", color="#98a2b3")))
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def plot_pr_curves(bundle: ModelBundle, selected_models: list[str]) -> go.Figure:
    fig = go.Figure()
    baseline = bundle.y_test.mean()
    for name in selected_models:
        probabilities = bundle.models[name].predict_proba(bundle.X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(bundle.y_test, probabilities)
        auc = average_precision_score(bundle.y_test, probabilities)
        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"{name} AP={auc:.3f}"))
    fig.add_hline(y=baseline, line_dash="dash", line_color="#98a2b3", annotation_text=f"Baseline={baseline:.3f}")
    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        template="plotly_white",
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def model_result_table(bundle: ModelBundle, row: dict, threshold: float, selected_models: list[str]) -> pd.DataFrame:
    input_df = row_to_frame(row)
    rows = []
    for name in selected_models:
        prob = predict_probability(bundle.models[name], input_df)
        label, distance = confidence_label(prob, threshold)
        rows.append(
            {
                "model": name,
                "probability_gt_50k": prob,
                "prediction": class_from_probability(prob, threshold),
                "confidence": label,
                "distance_from_threshold": distance,
            }
        )
    return pd.DataFrame(rows)


def page_home(df: pd.DataFrame, bundle: ModelBundle) -> None:
    st.title("Income Prediction and Workforce Analytics")
    st.markdown(
        """
        <div class="insight-card">
        This app predicts whether annual income exceeds $50K and explores how education,
        occupation, working hours, and demographic variables relate to income outcomes.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cleaned Records", f"{len(df):,}")
    c2.metric("Features", f"{len(FEATURE_COLUMNS)}")
    c3.metric(">50K Rate", f"{(df['income'].eq('>50k').mean()):.1%}")
    c4.metric("Final Model", "Random Forest")

    st.subheader("Dataset Explorer")
    f1, f2, f3 = st.columns(3)
    sex_filter = f1.multiselect("Sex", get_options(df, "sex"), default=get_options(df, "sex"))
    education_filter = f2.multiselect("Education", get_options(df, "education"))
    occupation_filter = f3.multiselect("Occupation", get_options(df, "occupation"))

    filtered = df.copy()
    if sex_filter:
        filtered = filtered[filtered["sex"].isin(sex_filter)]
    if education_filter:
        filtered = filtered[filtered["education"].isin(education_filter)]
    if occupation_filter:
        filtered = filtered[filtered["occupation"].isin(occupation_filter)]

    left, right = st.columns(2)
    with left:
        income_counts = filtered["income"].value_counts().reset_index()
        income_counts.columns = ["income", "count"]
        fig = px.bar(income_counts, x="income", y="count", color="income", color_discrete_sequence=["#2563eb", "#0f9f8f"])
        fig.update_layout(title="Income Distribution", template="plotly_white", showlegend=False, height=360)
        st.plotly_chart(fig, use_container_width=True)

        fig = px.histogram(filtered, x="age", color="income", nbins=30, barmode="overlay", color_discrete_sequence=["#2563eb", "#0f9f8f"])
        fig.update_layout(title="Age Distribution by Income", template="plotly_white", height=360)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        edu_rate = (
            filtered.assign(high_income=filtered["income"].eq(">50k").astype(int))
            .groupby("education", as_index=False)
            .agg(high_income_rate=("high_income", "mean"), n=("high_income", "size"))
            .sort_values("high_income_rate", ascending=False)
        )
        fig = px.bar(edu_rate, y="education", x="high_income_rate", color="high_income_rate", color_continuous_scale="Teal")
        fig.update_layout(title="High-Income Rate by Education", template="plotly_white", height=360, coloraxis_showscale=False)
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        occ_rate = (
            filtered.assign(high_income=filtered["income"].eq(">50k").astype(int))
            .groupby("occupation", as_index=False)
            .agg(high_income_rate=("high_income", "mean"), n=("high_income", "size"))
            .query("n >= 50")
            .sort_values("high_income_rate", ascending=False)
        )
        fig = px.bar(occ_rate, y="occupation", x="high_income_rate", color="high_income_rate", color_continuous_scale="Blues")
        fig.update_layout(title="High-Income Rate by Occupation", template="plotly_white", height=360, coloraxis_showscale=False)
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        <div class="warning-card">
        The dataset includes sensitive demographic attributes. Results should be treated as
        educational analysis, not as a real hiring, salary, or credit decision tool.
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_prediction(df: pd.DataFrame, bundle: ModelBundle) -> None:
    st.title("Prediction Center")
    st.caption("Choose prediction mode, model, and threshold before running individual or batch predictions.")

    c1, c2, c3 = st.columns([1.1, 1.1, 1.4])
    mode = c1.radio("Prediction Mode", ["Individual Prediction", "Batch Prediction"], horizontal=False)
    model_name = c2.selectbox("Primary Model", list(bundle.models.keys()), index=1)
    threshold = c3.slider("Classification Threshold", 0.10, 0.90, 0.50, 0.01)
    compare_all = st.toggle("Compare all models in the prediction output", value=True)
    selected_models = list(bundle.models.keys()) if compare_all else [model_name]

    if mode == "Individual Prediction":
        st.subheader("Individual Input")
        row = input_form(df, "predict")
        run = st.button("Run Prediction", use_container_width=True)
        if run:
            st.session_state["last_input"] = row
            result = model_result_table(bundle, row, threshold, selected_models)
            primary_result = result[result["model"] == model_name].iloc[0]

            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Prediction", primary_result["prediction"])
            m2.metric("P(>50K)", f"{primary_result['probability_gt_50k']:.1%}")
            m3.metric("Confidence", primary_result["confidence"])
            m4.metric("Threshold", f"{threshold:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)

            st.subheader("Model Comparison")
            display_result = result.copy()
            display_result["probability_gt_50k"] = display_result["probability_gt_50k"].map(lambda x: f"{x:.1%}")
            display_result["distance_from_threshold"] = display_result["distance_from_threshold"].map(lambda x: f"{x:.3f}")
            st.dataframe(display_result, use_container_width=True, hide_index=True)

            st.subheader("Prediction Explanation")
            explanation = local_explanation(bundle.models[model_name], row, bundle.baseline_row)
            e1, e2 = st.columns([1.2, 1])
            with e1:
                st.plotly_chart(plot_local_explanation(explanation), use_container_width=True)
            with e2:
                st.dataframe(
                    explanation[["feature", "current_value", "baseline_value", "contribution"]].style.format(
                        {"contribution": "{:.3f}"}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                top_features = ", ".join(explanation["feature"].head(5).tolist())
                st.markdown(
                    f"""
                    <div class="insight-card">
                    The prediction is mainly influenced by {top_features}. Positive bars increase
                    the predicted probability of >50K, while negative bars decrease it.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    else:
        st.subheader("Batch CSV Upload")
        st.markdown('<p class="small-muted">The CSV must contain the same feature columns used by the model.</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            batch = pd.read_csv(uploaded)
            missing = [col for col in FEATURE_COLUMNS if col not in batch.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
                batch = normalize_batch(batch)
                st.write("Preview")
                st.dataframe(batch.head(20), use_container_width=True)
                if st.button("Run Batch Prediction", use_container_width=True):
                    output = batch.copy()
                    models_for_batch = selected_models
                    for name in models_for_batch:
                        probabilities = bundle.models[name].predict_proba(batch[FEATURE_COLUMNS])[:, 1]
                        output[f"{name}_probability_gt_50k"] = probabilities
                        output[f"{name}_prediction"] = np.where(probabilities >= threshold, ">50K", "<=50K")

                    st.subheader("Prediction Results")
                    st.dataframe(output.head(100), use_container_width=True)

                    primary_prob_col = f"{model_name}_probability_gt_50k"
                    primary_pred_col = f"{model_name}_prediction"
                    b1, b2, b3 = st.columns(3)
                    b1.metric("Rows Scored", f"{len(output):,}")
                    b2.metric("Predicted >50K Rate", f"{(output[primary_pred_col].eq('>50K').mean()):.1%}")
                    b3.metric("Average P(>50K)", f"{output[primary_prob_col].mean():.1%}")

                    summary_cols = st.columns(2)
                    with summary_cols[0]:
                        edu_summary = output.groupby("education", as_index=False)[primary_prob_col].mean().sort_values(primary_prob_col, ascending=False)
                        fig = px.bar(edu_summary, y="education", x=primary_prob_col, color=primary_prob_col, color_continuous_scale="Teal")
                        fig.update_layout(title="Average Predicted Probability by Education", template="plotly_white", height=360, coloraxis_showscale=False)
                        fig.update_xaxes(tickformat=".0%")
                        st.plotly_chart(fig, use_container_width=True)
                    with summary_cols[1]:
                        sex_summary = output.groupby("sex", as_index=False)[primary_prob_col].mean().sort_values(primary_prob_col, ascending=False)
                        fig = px.bar(sex_summary, x="sex", y=primary_prob_col, color="sex", color_discrete_sequence=["#2563eb", "#0f9f8f"])
                        fig.update_layout(title="Average Predicted Probability by Sex", template="plotly_white", height=360, showlegend=False)
                        fig.update_yaxes(tickformat=".0%")
                        st.plotly_chart(fig, use_container_width=True)

                    csv = output.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Prediction Results",
                        data=csv,
                        file_name="adult_income_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )


def normalize_batch(batch: pd.DataFrame) -> pd.DataFrame:
    result = batch.copy()
    for col in CATEGORICAL_FEATURES:
        result[col] = result[col].astype(str).str.strip().str.lower()
    for col in NUMERIC_FEATURES + BINARY_FEATURES:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    result[NUMERIC_FEATURES + BINARY_FEATURES] = result[NUMERIC_FEATURES + BINARY_FEATURES].fillna(0)
    result[BINARY_FEATURES] = result[BINARY_FEATURES].astype(int)
    return result[FEATURE_COLUMNS]


def page_what_if(df: pd.DataFrame, bundle: ModelBundle) -> None:
    st.title("What-if Analysis")
    st.caption("Modify selected features and observe how the predicted probability changes.")

    c1, c2 = st.columns([1, 2])
    model_name = c1.selectbox("Model", list(bundle.models.keys()), index=1, key="whatif_model")
    threshold = c2.slider("Threshold", 0.10, 0.90, 0.50, 0.01, key="whatif_threshold")

    defaults = st.session_state.get("last_input", None)
    st.subheader("Base Profile")
    base_row = input_form(df, "whatif_base", defaults=defaults)

    st.subheader("Scenario Changes")
    s1, s2, s3, s4 = st.columns(4)
    scenario_education = s1.selectbox("Scenario Education", get_options(df, "education"), index=get_default_index(df, "education", "bachelors"))
    scenario_occupation = s2.selectbox("Scenario Occupation", get_options(df, "occupation"), index=get_default_index(df, "occupation", "prof-specialty"))
    scenario_hours = s3.slider("Scenario Hours per Week", int(df["hours_per_week"].min()), int(df["hours_per_week"].max()), 50)
    scenario_marital = s4.selectbox("Scenario Marital Status", get_options(df, "marital_status"), index=get_default_index(df, "marital_status", "married-civ-spouse"))

    scenarios = [
        ("Original", "None", base_row),
        ("Education Change", f"education = {scenario_education}", {**base_row, "education": scenario_education}),
        ("Occupation Change", f"occupation = {scenario_occupation}", {**base_row, "occupation": scenario_occupation}),
        ("Hours Change", f"hours_per_week = {scenario_hours}", {**base_row, "hours_per_week": int(scenario_hours)}),
        ("Marital Change", f"marital_status = {scenario_marital}", {**base_row, "marital_status": scenario_marital}),
        (
            "Combined Scenario",
            "education + occupation + hours + marital",
            {
                **base_row,
                "education": scenario_education,
                "occupation": scenario_occupation,
                "hours_per_week": int(scenario_hours),
                "marital_status": scenario_marital,
            },
        ),
    ]

    if st.button("Run What-if Simulation", use_container_width=True):
        model = bundle.models[model_name]
        original_prob = predict_probability(model, row_to_frame(base_row))
        rows = []
        for scenario_name, changed_feature, scenario_row in scenarios:
            prob = predict_probability(model, row_to_frame(scenario_row))
            rows.append(
                {
                    "scenario": scenario_name,
                    "changed_feature": changed_feature,
                    "probability_gt_50k": prob,
                    "prediction": class_from_probability(prob, threshold),
                    "change_vs_original": prob - original_prob,
                }
            )
        result = pd.DataFrame(rows)

        st.dataframe(
            result.style.format({"probability_gt_50k": "{:.1%}", "change_vs_original": "{:+.1%}"}),
            use_container_width=True,
            hide_index=True,
        )

        fig = px.bar(
            result,
            x="scenario",
            y="probability_gt_50k",
            color="change_vs_original",
            color_continuous_scale=["#e11d48", "#f8fafc", "#0f9f8f"],
        )
        fig.add_hline(y=threshold, line_dash="dash", line_color="#17212b", annotation_text=f"Threshold {threshold:.2f}")
        fig.update_layout(title="Scenario Probability Comparison", template="plotly_white", height=420, coloraxis_colorbar_title="Change")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        strongest = result.iloc[result["change_vs_original"].abs().argmax()]
        st.markdown(
            f"""
            <div class="insight-card">
            The largest simulated change is <b>{strongest['scenario']}</b>
            ({strongest['change_vs_original']:+.1%}). This is an association-based simulation,
            not a causal recommendation.
            </div>
            """,
            unsafe_allow_html=True,
        )


def page_performance(bundle: ModelBundle) -> None:
    st.title("Model Performance Dashboard")
    st.caption("Review model training results, threshold behavior, and curve-based evaluation.")

    display_mode = st.radio("Display Mode", ["Single Model", "Compare Two Models"], horizontal=True)
    threshold = st.slider("Evaluation Threshold", 0.10, 0.90, 0.50, 0.01, key="perf_threshold")

    if display_mode == "Single Model":
        model_name = st.selectbox("Model", list(bundle.models.keys()), index=1, key="single_perf_model")
        probabilities = bundle.models[model_name].predict_proba(bundle.X_test)[:, 1]
        metrics = prediction_metrics(bundle.y_test, probabilities, threshold)
        metric_cards(metrics)

        st.markdown(f'<div class="insight-card">{MODEL_NOTES[model_name]}</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_confusion(bundle.y_test, probabilities, threshold, f"{model_name} Confusion Matrix"), use_container_width=True)
        with c2:
            fp_fn = pd.DataFrame(
                {
                    "error_type": ["False Positive", "False Negative"],
                    "count": [metrics["fp"], metrics["fn"]],
                }
            )
            fig = px.bar(fp_fn, x="error_type", y="count", color="error_type", color_discrete_sequence=["#b7791f", "#e11d48"])
            fig.update_layout(title="Threshold Error Types", template="plotly_white", height=390, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        r1, r2 = st.columns(2)
        with r1:
            st.plotly_chart(plot_roc_curves(bundle, [model_name]), use_container_width=True)
        with r2:
            st.plotly_chart(plot_pr_curves(bundle, [model_name]), use_container_width=True)

        st.markdown(
            """
            <div class="warning-card">
            Lower thresholds usually increase recall but create more false positives.
            Higher thresholds usually improve precision but miss more actual high-income cases.
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        selected = st.multiselect(
            "Select up to two models",
            list(bundle.models.keys()),
            default=["Random Forest", "XGBoost"],
        )
        if len(selected) == 0:
            st.info("Select at least one model.")
            return
        if len(selected) > 2:
            st.warning("Only the first two selected models are shown for a clean comparison.")
            selected = selected[:2]

        compare_rows = []
        for name in selected:
            probabilities = bundle.models[name].predict_proba(bundle.X_test)[:, 1]
            row = prediction_metrics(bundle.y_test, probabilities, threshold)
            row["model"] = name
            compare_rows.append(row)
        compare_df = pd.DataFrame(compare_rows)
        st.dataframe(
            compare_df[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "fp", "fn"]].style.format(
                {
                    "accuracy": "{:.3f}",
                    "precision": "{:.3f}",
                    "recall": "{:.3f}",
                    "f1": "{:.3f}",
                    "roc_auc": "{:.3f}",
                    "pr_auc": "{:.3f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        long_df = compare_df.melt(
            id_vars="model",
            value_vars=["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"],
            var_name="metric",
            value_name="score",
        )
        fig = px.bar(long_df, x="metric", y="score", color="model", barmode="group", color_discrete_sequence=["#2563eb", "#0f9f8f"])
        fig.update_layout(title="Metric Comparison", template="plotly_white", height=420)
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_roc_curves(bundle, selected), use_container_width=True)
        with c2:
            st.plotly_chart(plot_pr_curves(bundle, selected), use_container_width=True)

        cm_cols = st.columns(len(selected))
        for col, name in zip(cm_cols, selected):
            probabilities = bundle.models[name].predict_proba(bundle.X_test)[:, 1]
            with col:
                st.plotly_chart(plot_confusion(bundle.y_test, probabilities, threshold, f"{name}"), use_container_width=True)

        best_f1 = compare_df.sort_values("f1", ascending=False).iloc[0]
        best_auc = compare_df.sort_values("roc_auc", ascending=False).iloc[0]
        st.markdown(
            f"""
            <div class="insight-card">
            At threshold {threshold:.2f}, <b>{best_f1['model']}</b> has the highest F1-score
            among the selected models. <b>{best_auc['model']}</b> has the highest ROC-AUC,
            which reflects ranking ability across thresholds.
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    inject_css()
    first_load = not st.session_state.get("_initial_load_complete", False)
    loading_placeholder = st.empty()
    if first_load:
        loading_placeholder.markdown(
            loading_ring("Loading dataset and training models..."),
            unsafe_allow_html=True,
        )
    df = load_data()
    bundle = train_models(df)
    if first_load:
        loading_placeholder.empty()
        st.session_state["_initial_load_complete"] = True

    pages = [
        "Home / Dataset Explorer",
        "Prediction Center",
        "What-if Analysis",
        "Model Performance",
    ]
    if "page" not in st.session_state:
        st.session_state["page"] = pages[0]

    st.sidebar.title("Navigation")
    for page_name in pages:
        if st.session_state["page"] == page_name:
            st.sidebar.markdown(f'<div class="nav-active">{page_name}</div>', unsafe_allow_html=True)
        else:
            if st.sidebar.button(page_name, key=f"nav_{page_name}", use_container_width=True):
                st.session_state["page"] = page_name
                st.rerun()
    page = st.session_state["page"]

    st.sidebar.markdown("---")
    st.sidebar.caption("Adult Income Dataset | Academic demo")
    if not XGBOOST_AVAILABLE:
        st.sidebar.info("xgboost is not installed. XGBoost page uses HistGradientBoosting fallback.")

    if page == "Home / Dataset Explorer":
        page_home(df, bundle)
    elif page == "Prediction Center":
        page_prediction(df, bundle)
    elif page == "What-if Analysis":
        page_what_if(df, bundle)
    else:
        page_performance(bundle)


if __name__ == "__main__":
    main()
