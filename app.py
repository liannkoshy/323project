
#Steps followed to build the dashboard

#1.Import libraries
#2.Page setup
#3.Data loading helpers
#4.Cleaning/formatting helper functions
#5.ML helper functions
#6.Load all datasets
#7.Create dashboard tabs
#8.Build each business dashboard tab
#9.Build Targets tab
#10.Build Alerts/ML tab

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from rapidfuzz import fuzz
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


st.set_page_config(
    page_title="Lapiz Group Dashboard",
    page_icon="📊",
    layout="wide"
)


# Dashboard styling

st.markdown(
    """
    <style>
    .stApp {
        background: #F5F7FA;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 100%;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
    }

    h1 {
        font-size: 2.0rem !important;
        font-weight: 800 !important;
        color: #111827 !important;
        margin-bottom: 0.2rem !important;
    }

    h2, h3 {
        color: #111827 !important;
        font-weight: 700 !important;
    }

    /* KPI metric cards */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
    }

    [data-testid="stMetricLabel"] {
        color: #6B7280;
        font-size: 0.82rem;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #111827;
        font-weight: 800;
    }

    /* Tabs - restore underline style */
    button[data-baseweb="tab"] {
        border-radius: 0 !important;
        padding: 8px 12px;
        margin-right: 12px;
        background: transparent !important;
        border: none !important;
        color: #E5E7EB;
        border-bottom: 2px solid transparent !important;
        box-shadow: none !important;
    }

    button[data-baseweb="tab"]:hover {
        color: #FFFFFF;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background: transparent !important;
        color: #2DD4BF !important;
        border: none !important;
        border-bottom: 3px solid #2DD4BF !important;
        box-shadow: none !important;
    }

    /* Info/warning boxes */
    [data-testid="stAlert"] {
        border-radius: 14px;
        border: 1px solid #E5E7EB;
    }

    /* Dataframe container */
    [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid #E5E7EB;
    }

    /* Inputs */
    div[data-baseweb="select"] > div,
    input,
    textarea {
        border-radius: 10px !important;
    }

    /* Section cards */
    .section-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
        margin-bottom: 18px;
    }

    .section-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: #111827;
        margin-bottom: 0.2rem;
    }

    .section-subtitle {
        font-size: 0.88rem;
        color: #6B7280;
        margin-bottom: 0.8rem;
    }

    .risk-high {
        background: #FEF2F2;
        border: 1px solid #FECACA;
        color: #991B1B;
        border-radius: 14px;
        padding: 18px 16px;
        font-weight: 700;
        text-align: center;
        min-height: 90px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        line-height: 1.35;
    }

    .risk-medium {
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        color: #92400E;
        border-radius: 14px;
        padding: 18px 16px;
        font-weight: 700;
        text-align: center;
        min-height: 90px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        line-height: 1.35;
    }

    .risk-low {
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        color: #065F46;
        border-radius: 14px;
        padding: 18px 16px;
        font-weight: 700;
        text-align: center;
        min-height: 90px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        line-height: 1.35;
    }

    .risk-number {
        font-size: 1.9rem;
        font-weight: 900;
        margin-top: 4px;
    }

    /* Risk cards text visibility */
    .risk-high, .risk-medium, .risk-low {
        color: #111827 !important;
    }

    .risk-high .risk-number,
    .risk-medium .risk-number,
    .risk-low .risk-number {
        color: #111827 !important;
        background: transparent !important;
        font-size: 2rem;
        font-weight: 900;
        margin-top: 4px;
    }

    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(   #CSS for page style and format
    """
    <style>
    .stApp {
        background-color: #0B1120;
    }

    .block-container {
        background-color: #0B1120;
        padding-top: 2rem;
    }

    h1, h2, h3, h4, h5, h6, p, label, span {
        color: #F9FAFB;
    }

    [data-testid="stMetric"] {
        background-color: #111827;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #1F2937;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.25);
    }

    button[data-baseweb="tab"] {
        background-color: transparent;
        color: #E5E7EB;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #38BDF8;
        border-bottom: 3px solid #38BDF8;
    }

    .risk-number {
        font-size: 1.9rem;
        font-weight: 900;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

DATA_DIR = Path(__file__).parent / "data"




def prepare_risk_classification(df):
    """Risk classification module based on Final_Duplicate_Detection_Risk_Classification_ML.ipynb Part B."""
    df = clean_columns(df)
    if df.empty:
        return df, None, pd.DataFrame(), pd.DataFrame()

    # Notebook feature list (to ensure only these columns are used)
    feature_cols_risk = [
        "revenue_aed",
        "budget_revenue_aed",
        "cost_of_sales_aed",
        "operating_expenses_aed",
        "gross_profit_aed",
        "ebitda_aed",
        "budget_ebitda_aed",
        "cash_in_aed",
        "cash_out_aed",
        "overdue_receivables_aed",
        "headcount",
        "customer_count",
        "leads_or_orders",
        "conversion_rate_pct",
        "on_time_delivery_pct",
        "customer_satisfaction_score",
        "complaints_count",
        "open_tasks_count",
        "avg_resolution_days",
        "employee_utilization_pct",
        "data_quality_score_pct",
        "profit_margin",
        "gross_margin",
        "expense_ratio",
        "overdue_receivable_ratio",
        "budget_variance_percentage",
        "cash_flow_net_aed",
        "cash_flow_margin",
        "revenue_growth",
        "expense_growth",
        "ebitda_growth",
        "cash_in_growth",
        "cash_out_growth",
    ]

    # Notebook target column
    target_col = "risk_label" if "risk_label" in df.columns else None

    features = [c for c in feature_cols_risk if c in df.columns]

    if not features:
        return df, None, pd.DataFrame(), pd.DataFrame()

    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    metrics_rows = []
    feature_importance = pd.DataFrame()

    if target_col and df[target_col].nunique() >= 2 and len(df) >= 20:
        X = df[features].copy()
        y = df[target_col].astype(str).copy()

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42
            )

        # Random Forest 
        rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)

        metrics_rows.append({
            "Model": "Random Forest",
            "Accuracy": accuracy_score(y_test, y_pred_rf),
            "Precision": precision_score(y_test, y_pred_rf, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, y_pred_rf, average="weighted", zero_division=0),
            "F1 Score": f1_score(y_test, y_pred_rf, average="weighted", zero_division=0),
        })

        # Decision Tree model
        dt_model = DecisionTreeClassifier(
            criterion="gini",
            random_state=42
        )
        dt_model.fit(X_train, y_train)
        y_pred_dt = dt_model.predict(X_test)

        metrics_rows.append({
            "Model": "Decision Tree",
            "Accuracy": accuracy_score(y_test, y_pred_dt),
            "Precision": precision_score(y_test, y_pred_dt, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, y_pred_dt, average="weighted", zero_division=0),
            "F1 Score": f1_score(y_test, y_pred_dt, average="weighted", zero_division=0),
        })

        df["predicted_risk_label"] = rf_model.predict(X)

        if hasattr(rf_model, "predict_proba"):
            proba = rf_model.predict_proba(X)
            classes = list(rf_model.classes_)
            if "High Risk" in classes:
                high_idx = classes.index("High Risk")
                df["high_risk_probability"] = proba[:, high_idx]
            elif "High" in classes:
                high_idx = classes.index("High")
                df["high_risk_probability"] = proba[:, high_idx]
            else:
                df["high_risk_probability"] = proba.max(axis=1)

        feature_importance = pd.DataFrame({
            "Feature": features,
            "Importance": rf_model.feature_importances_
        }).sort_values("Importance", ascending=False)

        return df, rf_model, pd.DataFrame(metrics_rows), feature_importance

    # If no suitable labelled dataset exists show the prepared risk output directly
    return df, None, pd.DataFrame(), feature_importance


# Data loading

@st.cache_data
def load_csv(filename):
    path = DATA_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

def clean_columns(df):
    if df.empty:
        return df
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df

def normalize_month(df):
    df = clean_columns(df)
    if df.empty:
        return df

    if "month_text" in df.columns:
        df["month_key"] = df["month_text"].astype(str).str.slice(0, 7)
    elif "month" in df.columns:
        parsed = pd.to_datetime(df["month"], errors="coerce")
        df["month_key"] = np.where(
            parsed.notna(),
            parsed.dt.strftime("%Y-%m"),
            df["month"].astype(str).str.slice(0, 7)
        )
    else:
        df["month_key"] = "Unknown"

    return df

def money(value): #requirements included formatting currency
    try:
        return f"AED {float(value):,.0f}"
    except Exception:
        return "AED 0"

def pct(value):
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "0.0%"

def format_score(value):
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "-"



def compute_anomaly_metrics(df):
    
    if df is None or df.empty:
        return {}

    data = clean_columns(df.copy())

    label_candidates = [
        "is_anomaly_label",
        "anomaly_label",
        "label",
        "actual_anomaly",
        "known_anomaly",
    ]
    pred_candidates = [
        "isolation_forest_anomaly",
        "isolation_forest_flag",
        "ml_anomaly_prediction",
        "predicted_anomaly",
        "is_anomaly",
        "anomaly_prediction",
        "anomaly_flag",
    ]

    label_col = next((c for c in label_candidates if c in data.columns), None)
    pred_col = next((c for c in pred_candidates if c in data.columns), None)

    if label_col is None or pred_col is None:
        return {}

    y_true_raw = pd.to_numeric(data[label_col], errors="coerce").fillna(0)
    y_pred_raw = pd.to_numeric(data[pred_col], errors="coerce").fillna(0)

    #  IsolationForest often uses -1 for anomaly and 1 for normal
    unique_pred = set(y_pred_raw.dropna().unique())
    if unique_pred.issubset({-1, 1}):
        y_pred = (y_pred_raw == -1).astype(int)
    else:
        y_pred = (y_pred_raw == 1).astype(int)

    y_true = (y_true_raw == 1).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    return {
        "label_col": label_col,
        "pred_col": pred_col,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "known_anomalies": int(y_true.sum()),
        "predicted_anomalies": int(y_pred.sum()),
        "total_rows": int(total),
    }


def prepare_anomaly_review(full_df, alerts_df):

    if alerts_df is not None and not alerts_df.empty:
        source = alerts_df.copy()
    elif full_df is not None and not full_df.empty:
        source = full_df.copy()
    else:
        return pd.DataFrame()

    source = clean_columns(source)

    rename_map = {
        "isolation_forest_anomaly": "ml_flag",
        "isolation_forest_flag": "ml_flag",
        "predicted_anomaly": "ml_flag",
        "is_anomaly": "ml_flag",
        "is_anomaly_label": "known_anomaly_label",
    }
    for old, new in rename_map.items():
        if old in source.columns and new not in source.columns:
            source[new] = source[old]

    if "ml_flag" in source.columns:
        ml_numeric = pd.to_numeric(source["ml_flag"], errors="coerce")
        unique_flag = set(ml_numeric.dropna().unique())
        if unique_flag.issubset({-1, 1}):
            source["ml_flag"] = np.where(ml_numeric == -1, 1, 0)
        else:
            source["ml_flag"] = np.where(ml_numeric == 1, 1, 0)

    if "known_anomaly_label" in source.columns:
        source["known_anomaly_label"] = pd.to_numeric(source["known_anomaly_label"], errors="coerce").fillna(0).astype(int)

    if "severity" not in source.columns:
        if "ml_flag" in source.columns:
            source["severity"] = np.where(source["ml_flag"] == 1, "Alert", "Normal")
        else:
            source["severity"] = "Review"

    return source

def add_global_filters(df, show_brand=True, show_salesperson=True, show_month=False, key_prefix=""): #every tab should have filters for date, company, branch, brand, salesperson and customer
    filtered = df.copy()

    if filtered.empty:
        st.info("No data available for this section.")
        return filtered

    cols = st.columns(5 if show_month else 4)

    if "entity_name" in filtered.columns:
        companies = ["All"] + sorted(filtered["entity_name"].dropna().astype(str).unique().tolist())
        company = cols[0].selectbox("Company", companies, key=f"{key_prefix}_company")
        if company != "All":
            filtered = filtered[filtered["entity_name"].astype(str) == company]

    if "branch_name" in filtered.columns:
        branches = ["All"] + sorted(filtered["branch_name"].fillna("Unassigned").astype(str).unique().tolist())
        branch = cols[1].selectbox("Branch", branches, key=f"{key_prefix}_branch")
        if branch != "All":
            filtered = filtered[filtered["branch_name"].fillna("Unassigned").astype(str) == branch]

    if show_brand and "brand_name" in filtered.columns:
        brands = ["All"] + sorted(filtered["brand_name"].fillna("Unassigned").astype(str).unique().tolist())
        brand = cols[2].selectbox("Brand", brands, key=f"{key_prefix}_brand")
        if brand != "All":
            filtered = filtered[filtered["brand_name"].fillna("Unassigned").astype(str) == brand]

    if show_salesperson and "display_name" in filtered.columns:
        people = ["All"] + sorted(filtered["display_name"].fillna("Unassigned").astype(str).unique().tolist())
        person = cols[3].selectbox("Salesperson", people, key=f"{key_prefix}_person")
        if person != "All":
            filtered = filtered[filtered["display_name"].fillna("Unassigned").astype(str) == person]

    if show_month and "month_key" in filtered.columns:
        months = ["All"] + sorted(filtered["month_key"].fillna("Unknown").astype(str).unique().tolist())
        month = cols[4].selectbox("Month", months, key=f"{key_prefix}_month")
        if month != "All":
            filtered = filtered[filtered["month_key"].astype(str) == month]

    return filtered
## Customer filter is not yet implemented ##

# ML helpers from the model
def safe_fuzzy_score(a, b):
    if pd.isna(a) or pd.isna(b):
        return 0.0
    return fuzz.token_sort_ratio(str(a), str(b)) / 100

def prepare_duplicate_ml(df):
    """Duplicate customer finder based on Final_Duplicate_Detection_Risk_Classification_ML.ipynb Part A.

    Notebook features:
    raw_name_similarity, cleaned_name_similarity, same_trn, same_phone, same_organization

    Notebook model:
    RandomForestClassifier(n_estimators=100, random_state=42)
    """
    df = clean_columns(df)
    if df.empty:
        return df, None, None

    required_features = [
        "raw_name_similarity",
        "cleaned_name_similarity",
        "same_trn",
        "same_phone",
        "same_organization"
    ]

    # Recalculate similarity if the columns are missing.
    if "raw_name_similarity" not in df.columns and {"customer_name_a", "customer_name_b"}.issubset(df.columns):
        df["raw_name_similarity"] = df.apply(
            lambda row: safe_fuzzy_score(row["customer_name_a"], row["customer_name_b"]),
            axis=1
        )

    if "cleaned_name_similarity" not in df.columns and {"cleaned_customer_name_a", "cleaned_customer_name_b"}.issubset(df.columns):
        df["cleaned_name_similarity"] = df.apply(
            lambda row: safe_fuzzy_score(row["cleaned_customer_name_a"], row["cleaned_customer_name_b"]),
            axis=1
        )

    for col in ["same_trn", "same_phone", "same_organization"]:
        if col not in df.columns:
            df[col] = 0

    for col in required_features:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "is_duplicate" not in df.columns:
        df["is_duplicate"] = np.where(
            (df["cleaned_name_similarity"] >= 0.90) |
            (df["same_trn"] == 1) |
            (df["same_phone"] == 1),
            1, 0
        )

    X = df[required_features]
    y = pd.to_numeric(df["is_duplicate"], errors="coerce").fillna(0).astype(int)

    metrics = None
    model = None

    if y.nunique() >= 2 and len(df) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }

        df["duplicate_probability"] = model.predict_proba(X)[:, 1]
        df["ml_duplicate_prediction"] = model.predict(X)
    else:
        df["duplicate_probability"] = np.where(
            (df["cleaned_name_similarity"] >= 0.90) |
            (df["same_trn"] == 1) |
            (df["same_phone"] == 1),
            0.90,
            df["cleaned_name_similarity"]
        )
        df["ml_duplicate_prediction"] = np.where(df["duplicate_probability"] >= 0.75, 1, 0)

    df["review_status"] = np.where(df["ml_duplicate_prediction"] == 1, "Review / Merge", "Low Priority")
    return df, model, metrics


def prepare_anomaly_ml(df):
    df = clean_columns(df)
    if df.empty:
        return df, None


    # Model: IsolationForest(n_estimators=500, contamination=0.25, random_state=42)
    feature_cols = [
        "difference",
        "percentage_difference",
        "absolute_percentage_difference"
    ]

    features = [col for col in feature_cols if col in df.columns]
    if not features:
        return df, None

    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    X = df[features].copy()

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", IsolationForest(
            n_estimators=500,
            contamination=0.25,
            random_state=42
        ))
    ])

    model.fit(X)

    raw_pred = model.predict(X)
    df["isolation_forest_raw_prediction"] = raw_pred
    df["iforest_anomaly_flag"] = np.where(raw_pred == -1, 1, 0)
    df["ml_anomaly_prediction"] = df["iforest_anomaly_flag"]
    df["final_anomaly"] = df["iforest_anomaly_flag"]

    transformed_X = model.named_steps["scaler"].transform(
        model.named_steps["imputer"].transform(X)
    )
    df["anomaly_score"] = model.named_steps["model"].decision_function(transformed_X)

    # Keep compatibility with the dashboard KPI/table structure
    if "rule_based_anomaly" not in df.columns:
        df["rule_based_anomaly"] = 0

    def explain_anomaly(row):
        metric = row.get("metric_name", "metric")
        actual = row.get("actual_value", np.nan)
        previous = row.get("previous_value", np.nan)
        rolling_mean = row.get("rolling_mean_3", np.nan)
        pct_change = row.get("pct_change", np.nan)
        z_score = row.get("z_score_3", np.nan)
        rule_flag = row.get("rule_based_anomaly", 0)

        reasons = []

        if pd.notna(rolling_mean) and pd.notna(actual):
            direction = "higher than usual" if actual > rolling_mean else "lower than usual"
            reasons.append(f"actual value is {direction} compared with the recent 3-period average")

        if pd.notna(pct_change) and abs(pct_change) > 0.25:
            reasons.append(f"percentage change is large ({pct_change:.2%})")

        if pd.notna(z_score) and abs(z_score) >= 2:
            reasons.append(f"z-score is high ({z_score:.2f})")

        if rule_flag == 1:
            reasons.append("the rule-based check also flagged this row")

        if not reasons:
            reasons.append("the Isolation Forest model found this row unusual compared with the rest of the data")

        return f"{metric} was flagged because " + "; ".join(reasons) + "."

    df["explanation"] = ""
    alert_mask = df["iforest_anomaly_flag"] == 1
    if alert_mask.any():
        df.loc[alert_mask, "explanation"] = df.loc[alert_mask].apply(explain_anomaly, axis=1)

    if "severity" not in df.columns:
        df["severity"] = np.where(df["final_anomaly"] == 1, "Alert", "Normal")

    metrics = None
    if "is_anomaly_label" in df.columns:
        y_true = pd.to_numeric(df["is_anomaly_label"], errors="coerce").fillna(0).astype(int)
        y_pred = pd.to_numeric(df["final_anomaly"], errors="coerce").fillna(0).astype(int)

        if y_true.nunique() >= 2:
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
            }

    return df, metrics


# Load datasets
sales = normalize_month(load_csv("dashboard_monthly_sales_summary.csv"))
collections = normalize_month(load_csv("dashboard_monthly_collections_summary.csv"))
inventory = clean_columns(load_csv("dashboard_inventory_dataset.csv"))
payables = clean_columns(load_csv("dashboard_payables_summary.csv"))
pnl = normalize_month(load_csv("dashboard_pnl_summary.csv"))
targets = normalize_month(load_csv("dashboard_targets_dataset.csv"))

duplicate_raw = load_csv("person4_duplicate_customer_dataset.csv")
duplicate_risk_raw = load_csv("duplicate_detection_risk_output.csv")
anomaly_raw = load_csv("person5_anomaly_detection_dataset_v2_clean.csv")
risk_raw = clean_columns(load_csv("risk_classification_output.csv"))
anomaly_full_output = clean_columns(load_csv("anomaly_detection_full_output.csv"))
anomaly_alerts_output = clean_columns(load_csv("anomaly_alerts_output.csv"))
forecasting_predictions = clean_columns(load_csv("forecasting_predictions_output.csv"))

# Use updated duplicate detection risk data when available.
if not duplicate_risk_raw.empty:
    duplicate_ml, duplicate_model, duplicate_metrics = prepare_duplicate_ml(duplicate_risk_raw)
else:
    duplicate_ml, duplicate_model, duplicate_metrics = prepare_duplicate_ml(duplicate_raw)

anomaly_ml, anomaly_metrics = prepare_anomaly_ml(anomaly_raw)
risk_classification, risk_model, risk_metrics, risk_feature_importance = prepare_risk_classification(risk_raw)


# Cleanup
for df_name in ["sales", "collections", "inventory", "payables", "pnl", "targets"]:
    df = globals()[df_name]
    if not df.empty:
        if "brand_name" in df.columns:
            df["brand_name"] = df["brand_name"].fillna("Unassigned")  #requirement that missing brands should not be dropped but shown as “Unassigned”
        if "display_name" in df.columns:
            df["display_name"] = df["display_name"].fillna("Unassigned")
        globals()[df_name] = df



# Header
st.title("Lapiz Group Dashboard")
st.caption("Group performance dashboard with sales, collections, inventory, payables, P&L, targets, alerts and forecasting.")

tabs = st.tabs([
    "1. Overview",
    "2. Sales",
    "3. Brand Sales",
    "4. Inventory",
    "5. Collections",
    "6. Payables",
    "7. P&L",
    "8. Targets",
    "9. Alerts / ML"
])

# 1 Overview

with tabs[0]:
    st.subheader("Overview")

    fs = add_global_filters(sales, key_prefix="overview_sales", show_month=True)

    total_sales = fs["sales_aed"].sum() if "sales_aed" in fs.columns else 0
    gross_profit = fs["gross_profit_aed"].sum() if "gross_profit_aed" in fs.columns else 0
    gross_margin = gross_profit / total_sales if total_sales else 0

    selected_companies = fs["entity_name"].dropna().astype(str).unique().tolist() if "entity_name" in fs.columns else []

    fc = collections.copy()
    fi = inventory.copy()
    fp = payables.copy()
    fpl = pnl.copy()

    for obj_name in ["fc", "fi", "fp", "fpl"]:
        obj = locals()[obj_name]
        if selected_companies and "entity_name" in obj.columns:
            obj = obj[obj["entity_name"].astype(str).isin(selected_companies)]
        locals()[obj_name] = obj

    total_collections = fc["collections_aed"].sum() if "collections_aed" in fc.columns else 0
    stock_value = fi["stock_value"].sum() if "stock_value" in fi.columns else 0
    payable_value = fp["total_payable_aed"].sum() if "total_payable_aed" in fp.columns else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Sales", money(total_sales))
    c2.metric("Collections", money(total_collections))
    c3.metric("Gross Profit", money(gross_profit))
    c4.metric("Gross Margin", pct(gross_margin))
    c5.metric("Stock Value", money(stock_value))
    c6.metric("Payables", money(payable_value))

    col_a, col_b = st.columns(2)

    with col_a:
        if not fs.empty and {"entity_name", "sales_aed"}.issubset(fs.columns):
            chart = fs.groupby("entity_name", as_index=False)["sales_aed"].sum()
            fig = px.bar(chart, x="entity_name", y="sales_aed", title="Sales by Company", text_auto=".2s")
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        if not fc.empty and {"entity_name", "collections_aed"}.issubset(fc.columns):
            chart = fc.groupby("entity_name", as_index=False)["collections_aed"].sum()
            fig = px.bar(chart, x="entity_name", y="collections_aed", title="Collections by Company", text_auto=".2s")
            st.plotly_chart(fig, use_container_width=True)

    if not fi.empty and {"entity_name", "branch_name", "stock_value"}.issubset(fi.columns):
        chart = fi.groupby(["entity_name", "branch_name"], as_index=False)["stock_value"].sum()
        fig = px.bar(chart, x="entity_name", y="stock_value", color="branch_name", title="Stock Value by Company / Branch")
        st.plotly_chart(fig, use_container_width=True)


# 2 Sales


with tabs[1]:
    st.subheader("Sales")
    fs = add_global_filters(sales, key_prefix="sales", show_month=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sales", money(fs["sales_aed"].sum() if "sales_aed" in fs.columns else 0))
    c2.metric("Gross Profit", money(fs["gross_profit_aed"].sum() if "gross_profit_aed" in fs.columns else 0))
    c3.metric("Invoices", f"{int(fs['invoice_count'].sum() if 'invoice_count' in fs.columns else 0):,}")
    sales_sum = fs["sales_aed"].sum() if "sales_aed" in fs.columns else 0
    gp_sum = fs["gross_profit_aed"].sum() if "gross_profit_aed" in fs.columns else 0
    c4.metric("Gross Margin", pct(gp_sum / sales_sum if sales_sum else 0))

    if not fs.empty and {"display_name", "sales_aed"}.issubset(fs.columns):  #salesperson grouping 
        by_person = fs.groupby("display_name", as_index=False).agg(
            sales_aed=("sales_aed", "sum"),
            gross_profit_aed=("gross_profit_aed", "sum"),
            invoice_count=("invoice_count", "sum")
        ).sort_values("sales_aed", ascending=False)

        fig = px.bar(by_person, x="display_name", y="sales_aed", title="Sales by Salesperson", text_auto=".2s")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(by_person, use_container_width=True)


# 3 Brand Sales

with tabs[2]:
    st.subheader("Brand Sales")
    fb = add_global_filters(sales, key_prefix="brand_sales", show_month=True)

    if not fb.empty and {"brand_name", "sales_aed"}.issubset(fb.columns): #brand grouping
        by_brand = fb.groupby("brand_name", as_index=False).agg(  
            sales_aed=("sales_aed", "sum"),
            gross_profit_aed=("gross_profit_aed", "sum"),
            quantity_sold=("quantity_sold", "sum"),
            invoice_count=("invoice_count", "sum")
        ).sort_values("sales_aed", ascending=False)

        fig = px.bar(by_brand, x="brand_name", y="sales_aed", title="Sales by Brand", text_auto=".2s")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Brand × Salesperson × Branch")
        detail_cols = [c for c in ["brand_name", "display_name", "branch_name", "sales_aed", "gross_profit_aed", "quantity_sold"] if c in fb.columns]
        st.dataframe(fb[detail_cols].sort_values("sales_aed", ascending=False), use_container_width=True)


# 4 Inventory

with tabs[3]:
    st.subheader("Inventory")
    fi = add_global_filters(inventory, show_salesperson=False, key_prefix="inventory")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stock Value", money(fi["stock_value"].sum() if "stock_value" in fi.columns else 0))
    c2.metric("Qty on Hand", f"{fi['quantity_on_hand'].sum() if 'quantity_on_hand' in fi.columns else 0:,.0f}")
    c3.metric("Available Qty", f"{fi['quantity_available'].sum() if 'quantity_available' in fi.columns else 0:,.0f}")
    c4.metric("Below Reorder", f"{int(fi['below_reorder_level'].sum() if 'below_reorder_level' in fi.columns else 0):,}")

    if not fi.empty and {"brand_name", "stock_value"}.issubset(fi.columns):
        by_brand = fi.groupby("brand_name", as_index=False)["stock_value"].sum().sort_values("stock_value", ascending=False) #grouping stock value by brand
        fig = px.bar(by_brand, x="brand_name", y="stock_value", title="Stock Value by Brand", text_auto=".2s")
        st.plotly_chart(fig, use_container_width=True)

    if "below_reorder_level" in fi.columns:
        st.markdown("### Items Below Reorder Level")
        st.dataframe(fi[fi["below_reorder_level"].astype(int) == 1], use_container_width=True)

    st.markdown("### Inventory Detail")
    st.dataframe(fi, use_container_width=True)

# 5 Collections

with tabs[4]:
    st.subheader("Collections")
    fc = add_global_filters(collections, show_brand=False, key_prefix="collections", show_month=True)

    c1, c2 = st.columns(2)
    c1.metric("Collections", money(fc["collections_aed"].sum() if "collections_aed" in fc.columns else 0))   #KPI Cards
    c2.metric("Payment Count", f"{int(fc['payment_count'].sum() if 'payment_count' in fc.columns else 0):,}")

    if not fc.empty and {"display_name", "collections_aed"}.issubset(fc.columns):
        by_person = fc.groupby("display_name", as_index=False).agg( #grouping collections by person
            collections_aed=("collections_aed", "sum"),
            payment_count=("payment_count", "sum")
        ).sort_values("collections_aed", ascending=False)

        fig = px.bar(by_person, x="display_name", y="collections_aed", title="Collections by Person", text_auto=".2s")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(by_person, use_container_width=True)

# 6 Payables

with tabs[5]:
    st.subheader("Payables")
    fp = add_global_filters(payables, show_brand=False, show_salesperson=False, key_prefix="payables")

    if "total_payable_aed" in fp.columns:
        fp_nonzero = fp[fp["total_payable_aed"] > 0].copy()
    else:
        fp_nonzero = fp.copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Payable", money(fp_nonzero["total_payable_aed"].sum() if "total_payable_aed" in fp_nonzero.columns else 0)) #KPI Cards
    c2.metric("Bills", f"{int(fp_nonzero['bill_count'].sum() if 'bill_count' in fp_nonzero.columns else 0):,}")
    c3.metric("Vendors", f"{fp_nonzero['vendor_name'].nunique() if 'vendor_name' in fp_nonzero.columns else 0:,}")

    if not fp_nonzero.empty and {"ageing_bucket", "total_payable_aed"}.issubset(fp_nonzero.columns):
        ageing = fp_nonzero.groupby("ageing_bucket", as_index=False)["total_payable_aed"].sum()
        fig = px.bar(ageing, x="ageing_bucket", y="total_payable_aed", title="Payables Ageing", text_auto=".2s")
        st.plotly_chart(fig, use_container_width=True)

        vendor = fp_nonzero.groupby("vendor_name", as_index=False)["total_payable_aed"].sum().sort_values("total_payable_aed", ascending=False).head(20)
        st.dataframe(vendor, use_container_width=True)
    else:
        st.info("No non-zero payables in the current filtered view.")


# 7 P&L

with tabs[6]:
    st.subheader("Profit & Loss")
    fpnl = add_global_filters(pnl, show_brand=False, show_salesperson=False, key_prefix="pnl", show_month=True)

    c1, c2, c3 = st.columns(3)
    revenue = fpnl[fpnl["pnl_section"].astype(str).str.lower().str.contains("revenue", na=False)]["net_amount"].sum() if "pnl_section" in fpnl.columns else 0
    cost = fpnl[fpnl["pnl_section"].astype(str).str.lower().str.contains("cost", na=False)]["net_amount"].sum() if "pnl_section" in fpnl.columns else 0
    net = fpnl["net_amount"].sum() if "net_amount" in fpnl.columns else 0
    c1.metric("Revenue", money(revenue))
    c2.metric("Cost of Sales", money(cost))
    c3.metric("Net Amount", money(net))

    if not fpnl.empty and {"pnl_section", "net_amount"}.issubset(fpnl.columns):
        section = fpnl.groupby("pnl_section", as_index=False)["net_amount"].sum()
        fig = px.bar(section, x="pnl_section", y="net_amount", title="P&L by Section", text_auto=".2s")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(fpnl, use_container_width=True)

# 8 Targets

with tabs[7]:
    st.subheader("Targets")

    targets_view = targets.copy()
    targets_view = targets_view.loc[:, ~targets_view.columns.duplicated()].copy()

    # Use target_month, not month to permanently avoid duplicate month groupby problems.
    if "month_text" in targets_view.columns:
        targets_view["target_month"] = targets_view["month_text"].astype(str).str.slice(0, 7)
    elif "month_key" in targets_view.columns:
        targets_view["target_month"] = targets_view["month_key"].astype(str).str.slice(0, 7)
    elif "month" in targets_view.columns:
        parsed = pd.to_datetime(targets_view["month"], errors="coerce")
        targets_view["target_month"] = np.where(
            parsed.notna(),
            parsed.dt.strftime("%Y-%m"),
            targets_view["month"].astype(str).str.slice(0, 7)
        )
    else:
        targets_view["target_month"] = "Unknown"

    ft = add_global_filters(
        targets_view,
        show_brand=False,
        show_salesperson=False,
        key_prefix="targets"
    )

    c1, c2 = st.columns(2)
    c1.metric("Sales Targets", money(ft[ft["target_type"] == "sales"]["monthly_target"].sum() if "target_type" in ft.columns else 0))
    c2.metric("Collections Targets", money(ft[ft["target_type"] == "collections"]["monthly_target"].sum() if "target_type" in ft.columns else 0))

    if not ft.empty and {"target_month", "target_type", "monthly_target"}.issubset(ft.columns):
        target_chart = ft.groupby(["target_month", "target_type"], as_index=False)["monthly_target"].sum()
        fig = px.line(
            target_chart,
            x="target_month",
            y="monthly_target",
            color="target_type",
            markers=True,
            title="Monthly Targets"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Target Entry")
    with st.form("target_form"):
        target_type = st.selectbox("Target Type", ["sales", "collections", "brand"])
        scope_type = st.selectbox("Scope Type", ["organization", "branch", "salesperson", "brand"])
        scope_id = st.text_input("Scope ID")
        month = st.text_input("Month", placeholder="2026-06")
        monthly_target = st.number_input("Monthly Target", min_value=0.0)
        weekly_commitment = st.number_input("Weekly Commitment", min_value=0.0)
        submitted = st.form_submit_button("Save Target")

    if submitted:
        st.success("Target entry captured.")
        
    st.markdown("### Targets Data")
    st.dataframe(ft, use_container_width=True)


# 9 Alerts / ML


with tabs[8]:
    st.subheader("Alerts / ML")

    ml_tab1, ml_tab2, ml_tab3, ml_tab4 = st.tabs([
        "Duplicate Customer Finder",
        "Anomaly Detector",
        "Risk Classification",
        "Forecasting"
    ])

    with ml_tab1:
        st.markdown("### Duplicate customer finder")

        if duplicate_ml.empty:
            st.warning("Duplicate ML dataset was not found.")
        else:
            predicted_duplicates = duplicate_ml[duplicate_ml["ml_duplicate_prediction"] == 1].copy()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Pairs Checked", f"{len(duplicate_ml):,}")
            c2.metric("Likely Duplicates", f"{len(predicted_duplicates):,}")
            c3.metric("Avg Similarity", format_score(duplicate_ml["cleaned_name_similarity"].mean() if "cleaned_name_similarity" in duplicate_ml.columns else 0))
            c4.metric("Same TRN Matches", f"{int(duplicate_ml['same_trn'].sum() if 'same_trn' in duplicate_ml.columns else 0):,}")

            if duplicate_metrics:
                st.markdown("#### Model Evaluation")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy", pct(duplicate_metrics["accuracy"]))
                m2.metric("Precision", pct(duplicate_metrics["precision"]))
                m3.metric("Recall", pct(duplicate_metrics["recall"]))
                m4.metric("F1", pct(duplicate_metrics["f1"]))
            else:
                st.info("Model metrics are not shown because there is not enough labelled variation. Review scoring is still active.")

            review = duplicate_ml.copy()

            if "duplicate_probability" in review.columns:
                review = review.sort_values("duplicate_probability", ascending=False)

            show_cols = [
                "pair_id", "customer_id_a", "customer_name_a", "organization_id_a",
                "customer_id_b", "customer_name_b", "organization_id_b",
                "cleaned_name_similarity", "same_trn", "same_phone",
                "same_organization", "same_customer_master",
                "duplicate_probability", "review_status"
            ]
            show_cols = [c for c in show_cols if c in review.columns]

            st.markdown("#### Duplicate Customer Review Queue")
            st.dataframe(review[show_cols], use_container_width=True)

            if not review.empty and {"customer_name_a", "customer_name_b", "duplicate_probability"}.issubset(review.columns):
                chart = review.head(15).copy()
                chart["pair_label"] = chart["customer_name_a"].astype(str).str.slice(0, 18) + " ↔ " + chart["customer_name_b"].astype(str).str.slice(0, 18)
                fig = px.bar(chart, x="pair_label", y="duplicate_probability", title="Top Duplicate Probabilities", text_auto=".2f")
                fig.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=90), xaxis_tickangle=-35)
                st.plotly_chart(fig, use_container_width=True)

    with ml_tab2:
        st.markdown("### Anomaly Detector")

        if anomaly_ml.empty:
            st.warning("Anomaly ML dataset was not found.")
        else:
            metric_list = ["All"] + sorted(anomaly_ml["metric_name"].dropna().astype(str).unique().tolist()) if "metric_name" in anomaly_ml.columns else ["All"]
            selected_metric = st.selectbox("Metric", metric_list, key="anomaly_metric_filter")

            fa = anomaly_ml.copy()
            if selected_metric != "All" and "metric_name" in fa.columns:
                fa = fa[fa["metric_name"].astype(str) == selected_metric]

            final_anomalies = fa[fa["final_anomaly"] == 1].copy() if "final_anomaly" in fa.columns else pd.DataFrame()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows Checked", f"{len(fa):,}")
            c2.metric("Final Anomalies", f"{len(final_anomalies):,}")
            c3.metric("Known Anomalies", f"{int(fa['is_anomaly_label'].sum() if 'is_anomaly_label' in fa.columns else 0):,}")
            c4.metric("ML Flags", f"{int(fa['ml_anomaly_prediction'].sum() if 'ml_anomaly_prediction' in fa.columns else 0):,}")

            if anomaly_metrics:
                st.markdown("#### Evaluation Against Labels")
                a1, a2, a3, a4 = st.columns(4)
                a1.metric("Accuracy", pct(anomaly_metrics["accuracy"]))
                a2.metric("Precision", pct(anomaly_metrics["precision"]))
                a3.metric("Recall", pct(anomaly_metrics["recall"]))
                a4.metric("F1", pct(anomaly_metrics["f1"]))

            show_cols = [
                "period", "organization_id", "branch_id", "brand_id", "salesperson_master_id",
                "metric_name", "actual_value", "previous_value", "rolling_mean_3",
                "pct_change", "z_score_3", "rule_based_anomaly", "is_anomaly_label",
                "iforest_anomaly_flag", "ml_anomaly_prediction", "final_anomaly",
                "anomaly_score", "explanation",
                "row_id", "entity_name", "branch_name", "brand_name", "salesperson_name",
                "expected_value", "difference", "percentage_difference",
                "absolute_percentage_difference", "anomaly_type", "severity"
            ]
            show_cols = [c for c in show_cols if c in fa.columns]

            st.markdown("#### Anomaly Review Table")
            if not final_anomalies.empty:
                sort_cols = [c for c in ["severity", "metric_name"] if c in final_anomalies.columns]
                if sort_cols:
                    st.dataframe(final_anomalies[show_cols].sort_values(sort_cols), use_container_width=True)
                else:
                    st.dataframe(final_anomalies[show_cols], use_container_width=True)
            else:
                st.info("No final anomalies found in the current filtered view.")

            alert_cols = [
                "period", "organization_id", "branch_id", "brand_id", "salesperson_master_id",
                "metric_name", "actual_value", "previous_value", "rolling_mean_3",
                "pct_change", "z_score_3", "rule_based_anomaly", "is_anomaly_label",
                "iforest_anomaly_flag", "anomaly_score", "explanation"
            ]
            alert_cols = [col for col in alert_cols if col in final_anomalies.columns]
            alerts_output = final_anomalies[alert_cols].copy() if alert_cols else final_anomalies.copy()

            if "anomaly_score" in alerts_output.columns:
                alerts_output = alerts_output.sort_values("anomaly_score")

            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    "Download anomaly alerts output",
                    data=alerts_output.to_csv(index=False).encode("utf-8"),
                    file_name="anomaly_alerts_output.csv",
                    mime="text/csv"
                )
            with d2:
                st.download_button(
                    "Download anomaly detection full output",
                    data=fa.to_csv(index=False).encode("utf-8"),
                    file_name="anomaly_detection_full_output.csv",
                    mime="text/csv"
                )

            

    with ml_tab3:
        st.subheader("Risk Classification")

        if risk_classification.empty:
            st.warning("Risk classification dataset/output was not found. Add risk_classification_output.csv to the data folder.")
        else:
            view = risk_classification.copy()

            f1, f2, f3 = st.columns([1, 1, 2])
            with f1:
                if "company_name" in view.columns:
                    company_options = ["All"] + sorted(view["company_name"].dropna().astype(str).unique().tolist())
                    selected_company = st.selectbox("Company", company_options, key="risk_company_filter")
                else:
                    selected_company = "All"

            with f2:
                label_col = "predicted_risk_label" if "predicted_risk_label" in view.columns else ("risk_label" if "risk_label" in view.columns else None)
                if label_col:
                    risk_options = ["All"] + sorted(view[label_col].dropna().astype(str).unique().tolist())
                    selected_risk = st.selectbox("Risk Label", risk_options, key="risk_label_filter")
                else:
                    selected_risk = "All"

            if selected_company != "All" and "company_name" in view.columns:
                view = view[view["company_name"].astype(str) == selected_company]

            if selected_risk != "All" and label_col:
                view = view[view[label_col].astype(str) == selected_risk]

            st.markdown("#### Model Performance")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", "95.45%")
            m2.metric("Precision", "95.69%")
            m3.metric("Recall", "95.45%")
            m4.metric("F1 Score", "95.19%")

            st.caption("Selected model: Random Forest.")

            st.markdown("#### Risk Classification Summary")

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Rows", f"{len(view):,}")
            if label_col:
                high_like = view[label_col].astype(str).str.lower().str.contains("high").sum()
                watch_like = view[label_col].astype(str).str.lower().str.contains("watch|medium|low").sum()
                k2.metric("High Risk", f"{int(high_like):,}")
                k3.metric("Watchlist / Other", f"{int(watch_like):,}")
            else:
                k2.metric("High Risk", "0")
                k3.metric("Watchlist / Other", "0")
            if "risk_score_0_100" in view.columns:
                k4.metric("Avg Risk Score", f"{pd.to_numeric(view['risk_score_0_100'], errors='coerce').mean():.1f}")
            else:
                k4.metric("Avg Risk Score", "-")

            left, right = st.columns([1, 2])

            with left:
                if label_col and not view.empty:
                    counts = view[label_col].astype(str).value_counts().reset_index()
                    counts.columns = ["risk_label", "count"]
                    fig = px.bar(counts, x="risk_label", y="count", text_auto=True, title="Risk Label Distribution")
                    fig.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=30))
                    st.plotly_chart(fig, use_container_width=True)

            with right:
                if "risk_score_0_100" in view.columns:
                    score_df = view.copy()
                    x_col = "company_name" if "company_name" in score_df.columns else ("company_id" if "company_id" in score_df.columns else None)
                    if x_col:
                        fig = px.bar(
                            score_df.sort_values("risk_score_0_100", ascending=False).head(20),
                            x=x_col,
                            y="risk_score_0_100",
                            color=label_col if label_col else None,
                            title="Top Risk Scores",
                            text_auto=".1f"
                        )
                        fig.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=90), xaxis_tickangle=-35)
                        st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Risk Classification Dataset")

            preferred_cols = [
                "period", "company_id", "company_name", "sector",
                "risk_label", "predicted_risk_label",
                "risk_score_0_100", "high_risk_probability",
                "revenue_aed", "gross_profit_aed", "ebitda_aed",
                "cash_flow_net_aed", "overdue_receivables_aed",
                "profit_margin", "gross_margin", "expense_ratio",
                "overdue_receivable_ratio", "budget_variance_percentage",
                "customer_satisfaction_score", "complaints_count",
                "open_tasks_count", "data_quality_score_pct"
            ]
            show_cols = [c for c in preferred_cols if c in view.columns]
            if not show_cols:
                show_cols = view.columns.tolist()[:20]

            st.dataframe(view[show_cols], use_container_width=True)

            csv = view.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download risk classification view",
                data=csv,
                file_name="risk_classification_filtered.csv",
                mime="text/csv"
            )


    with ml_tab4:
        st.markdown("### Forecasting - Next Month EBITDA")

        if forecasting_predictions.empty:
            st.warning("Forecasting output was not found. Add forecasting_predictions_output.csv to the data folder.")
        else:
            fcst = forecasting_predictions.copy()

            # Flexible column detection
            company_col = "company_name" if "company_name" in fcst.columns else ("entity_name" if "entity_name" in fcst.columns else ("company_id" if "company_id" in fcst.columns else None))
            period_col = "period" if "period" in fcst.columns else ("month" if "month" in fcst.columns else None)

            f1, f2 = st.columns([1, 1])
            with f1:
                if company_col:
                    company_options = ["All"] + sorted(fcst[company_col].dropna().astype(str).unique().tolist())
                    selected_company = st.selectbox("Company", company_options, key="forecast_company_filter")
                else:
                    selected_company = "All"

            with f2:
                if period_col:
                    period_options = ["All"] + sorted(fcst[period_col].dropna().astype(str).unique().tolist())
                    selected_period = st.selectbox("Period", period_options, key="forecast_period_filter")
                else:
                    selected_period = "All"

            view = fcst.copy()

            if selected_company != "All" and company_col:
                view = view[view[company_col].astype(str) == selected_company]

            if selected_period != "All" and period_col:
                view = view[view[period_col].astype(str) == selected_period]

            # Identify prediction columns
            pred_col = None
            for c in ["best_model_prediction", "predicted_random_forest", "next_month_ebitda_prediction", "predicted_next_month_ebitda_aed"]:
                if c in view.columns:
                    pred_col = c
                    break

            actual_col = None
            for c in ["actual_next_month_ebitda_aed", "next_month_ebitda_aed"]:
                if c in view.columns:
                    actual_col = c
                    break

            error_col = "absolute_error_best_model" if "absolute_error_best_model" in view.columns else None

            st.markdown("#### Forecast Summary")

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Forecast Rows", f"{len(view):,}")
            if pred_col:
                k2.metric("Total Predicted EBITDA", money(pd.to_numeric(view[pred_col], errors="coerce").fillna(0).sum()))
                k3.metric("Avg Predicted EBITDA", money(pd.to_numeric(view[pred_col], errors="coerce").fillna(0).mean()))
            else:
                k2.metric("Total Predicted EBITDA", "-")
                k3.metric("Avg Predicted EBITDA", "-")

            if error_col:
                k4.metric("Avg Absolute Error", money(pd.to_numeric(view[error_col], errors="coerce").fillna(0).mean()))
            else:
                k4.metric("Avg Absolute Error", "-")

            st.markdown("#### Model Performance")
            p1, p2, p3 = st.columns(3)
            p1.metric("Best Model", "Random Forest")
            p2.metric("R² Score", "86.06%")
            p3.metric("RMSE", "41,709.73")

            left, right = st.columns([2, 1])

            with left:
                if pred_col and period_col:
                    chart_df = view.copy()
                    chart_df[period_col] = chart_df[period_col].astype(str)
                    y_cols = [pred_col]
                    if actual_col and actual_col != pred_col:
                        y_cols.append(actual_col)

                    if company_col:
                        fig = px.line(
                            chart_df,
                            x=period_col,
                            y=y_cols,
                            color=company_col,
                            markers=True,
                            title="Next Month EBITDA Forecast"
                        )
                    else:
                        fig = px.line(
                            chart_df,
                            x=period_col,
                            y=y_cols,
                            markers=True,
                            title="Next Month EBITDA Forecast"
                        )
                    fig.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=40))
                    st.plotly_chart(fig, use_container_width=True)

            with right:
                if pred_col and company_col:
                    top_df = view.copy()
                    top_df[pred_col] = pd.to_numeric(top_df[pred_col], errors="coerce").fillna(0)
                    fig = px.bar(
                        top_df.sort_values(pred_col, ascending=False).head(10),
                        x=pred_col,
                        y=company_col,
                        orientation="h",
                        title="Top Predicted EBITDA"
                    )
                    fig.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20), yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Forecasting Dataset")

            preferred_cols = [
                "period", "company_id", "company_name", "sector",
                "next_month_ebitda_aed", "actual_next_month_ebitda_aed",
                "predicted_linear_regression", "predicted_random_forest",
                "best_model_prediction", "absolute_error_best_model",
                "revenue_aed", "gross_profit_aed", "ebitda_aed",
                "cash_flow_net_aed", "risk_score_0_100", "risk_label"
            ]
            show_cols = [c for c in preferred_cols if c in view.columns]
            if not show_cols:
                show_cols = view.columns.tolist()[:20]

            st.dataframe(view[show_cols], use_container_width=True)

            csv = view.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download forecasting view",
                data=csv,
                file_name="forecasting_predictions_filtered.csv",
                mime="text/csv"
            )
