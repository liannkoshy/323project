# Lapiz Group Dashboard

## Overview

This is a Streamlit dashboard for monitoring group-level business performance across sales, collections, inventory, payables, profit and loss, targets, alerts, risk classification, anomaly detection, and forecasting.

## Main Modules

- Overview
- Sales
- Brand Sales
- Inventory
- Collections
- Payables
- Profit & Loss
- Targets
- Alerts / ML
  - Duplicate Customer Finder
  - Anomaly Detector
  - Risk Classification
  - Forecasting

## How to Run

Open PowerShell or Command Prompt inside this folder and run:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If another Streamlit app is already running, use a different port:

```powershell
python -m streamlit run app.py --server.port 8507
```

## Data Folder

The dashboard reads CSV files from the `data` folder.

Important files include:

- dashboard_monthly_sales_summary.csv
- dashboard_monthly_collections_summary.csv
- dashboard_inventory_dataset.csv
- dashboard_payables_summary.csv
- dashboard_pnl_summary.csv
- dashboard_targets_dataset.csv
- person4_duplicate_customer_dataset.csv
- person5_anomaly_detection_dataset.csv
- duplicate_detection_risk_output.csv
- risk_classification_output.csv
- forecasting_predictions_output.csv

## Notes

The dashboard is read-only. It displays analytics based on the CSV files provided in the data folder.

The `__pycache__` folder is not required and can be deleted safely if it appears after running the app.


## Feature Importance Removed

The Risk Classification tab now focuses on business-facing risk labels, risk scores, summary counts, charts, and the risk table.

The feature importance chart has been removed to keep the dashboard simpler for business users.


## Anomaly Chart Fix

The Anomaly Detector trend chart has been adjusted so different value types are not mixed unnecessarily.

- When a specific metric is selected, the chart shows that metric only.
- When All is selected, the chart shows flagged anomaly points only.
- Period values are displayed as clean month labels where possible.


## Duplicate risk section removed

The Duplicate Customer Finder now focuses only on duplicate detection.

The High/Medium/Low risk cards, risk filter, risk chart, and risk-related columns were removed from the Duplicate Customer Finder.
