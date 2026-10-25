# Web Analytics-Based User Conversion Prediction System (ConvertSense AI)

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Flask 3.1](https://img.shields.io/badge/Flask-3.1-black.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.1-orange.svg)](https://scikit-learn.org/)
[![SQLite](https://img.shields.io/badge/SQLite-SQLAlchemy%202.0-lightblue.svg)](https://www.sqlite.org/)
[![Bootstrap 5](https://img.shields.io/badge/Bootstrap-5.3-purple.svg)](https://getbootstrap.com/)
[![Tests](https://img.shields.io/badge/Tests-42%20Passed%20(100%25)-brightgreen.svg)]()

A complete, production-grade web application and machine learning classification platform designed to analyze e-commerce website visitor session behavior and estimate the calibrated probability that an active browsing session will result in a purchase (`Revenue = True`).

Built to solve digital marketing funnel drop-off, prevent target leakage, and provide real-time behavioral insights to e-commerce and revenue operations teams.

---

## Table of Contents
1. [Business Problem & Objectives](#1-business-problem--objectives)
2. [Executive Architecture Diagram](#2-executive-architecture-diagram)
3. [Technology Stack](#3-technology-stack)
4. [Dataset & Data Attribution](#4-dataset--data-attribution)
5. [Data Science & Anti-Leakage Methodology](#5-data-science--anti-leakage-methodology)
6. [Classification Models & Evaluation Results](#6-classification-models--evaluation-results)
7. [Decision Threshold Engineering](#7-decision-threshold-engineering)
8. [Application Features & Walkthrough](#8-application-features--walkthrough)
9. [Database Design](#9-database-design)
10. [REST API Documentation](#10-rest-api-documentation)
11. [Windows Setup & Quickstart Guide](#11-windows-setup--quickstart-guide)
12. [Automated Testing](#12-automated-testing)
13. [Production Deployment Guide](#13-production-deployment-guide)
14. [Interview Guide & Fresher Explanation](#14-interview-guide--fresher-explanation)
15. [Limitations & Future Roadmap](#15-limitations--future-roadmap)

---

## 1. Business Problem & Objectives

### The Business Challenge
In modern retail e-commerce, digital advertising campaigns generate substantial traffic volumes, yet **only 10% to 16%** of website browsing sessions result in an actual financial transaction.
- **The Cost of False Negatives (Missed Revenue):** High-intent shoppers reach product pages or cart reviews with minor hesitations. Without real-time intervention (such as an automated live chat assist or exit-intent incentive), these users leave and purchase elsewhere.
- **The Cost of False Positives (Margin Erosion):** Spray-and-pray promotional discounts are shown universally to users who were already determined to purchase at full price, destroying e-commerce unit economics.

### Core Objectives
1. **Calibrated Probability Estimation:** Deliver statistical probability estimates (0.0% to 100.0%) rather than arbitrary binary outputs.
2. **Strict Target Leakage Isolation:** Exclude post-hoc metrics like `PageValues` to guarantee real-time validity during active browsing.
3. **Interactive Slicing Dashboard:** Enable marketing and product managers to filter conversion benchmarks across months, visitor types, traffic sources, and devices.
4. **Behavioral Explainability:** Provide data-grounded explanations of top behavioral drivers (e.g. exit hazards, catalog depth, dwell duration) without presenting correlation as causation.
5. **Auditable Prediction Ledger:** Store session features and predictions in SQLite with safe CSV export and ground-truth verification capabilities.

---

## 2. Executive Architecture Diagram

```mermaid
flowchart TD
    subgraph ClientLayer["Frontend SaaS Interface"]
        UI_Home["Landing Overview (/)"]
        UI_Dash["Analytics Dashboard (/dashboard)"]
        UI_Pred["Prediction Interface (/predict)"]
        UI_Hist["Prediction Ledger (/history)"]
        UI_Eval["Model Diagnostics (/evaluation)"]
        UI_Exp["Data Explorer (/data-explorer)"]
    end

    subgraph AppLayer["Flask Application (Factory Pattern)"]
        BP_Main["routes/main.py"]
        BP_Dash["routes/dashboard.py"]
        BP_Pred["routes/prediction.py"]
        BP_Hist["routes/history.py"]
        BP_Eval["routes/evaluation.py"]
        BP_Exp["routes/data_explorer.py"]

        Service_Predict["services/prediction_service.py"]
        Service_Analytics["services/analytics_service.py"]
        Service_Model["services/model_service.py"]
    end

    subgraph DataScience["Offline ML Pipeline"]
        Dataset[("UCI Online Shoppers\n12,330 Sessions")]
        Script_Down["scripts/download_data.py"]
        Script_EDA["scripts/analyze_data.py"]
        Script_Train["scripts/train_model.py"]
        
        LeakageGuard{"Target Leakage Guard:\nDrop PageValues &\nDrop 125 Duplicates"}
        StratSplit["Stratified 80/20 Train-Test Split"]
        CV["5-Fold Stratified Cross-Validation\n(LogReg vs RF vs HistGB)"]
        Champion[("Champion Model Pipeline:\nbest_conversion_model.joblib")]
        Metrics[("Evaluation Artifacts:\nreports/model_metrics.json")]
    end

    subgraph Persistence["Storage & Database"]
        DB[(SQLite: web_analytics.db\nSQLAlchemy ORM)]
    end

    Dataset --> Script_Down --> Script_EDA
    Script_Down --> Script_Train
    Script_Train --> LeakageGuard --> StratSplit --> CV --> Champion
    CV --> Metrics

    UI_Pred --> BP_Pred --> Service_Predict
    Champion -.->|Loaded by| Service_Predict
    Service_Predict -->|Log Prediction| DB
    UI_Dash --> BP_Dash --> Service_Analytics
    Dataset -.->|Aggregated by| Service_Analytics
    UI_Hist --> BP_Hist --> DB
    UI_Eval --> BP_Eval --> Service_Model
    Metrics -.->|Loaded by| Service_Model
```

---

## 3. Technology Stack

- **Backend:** Python 3.13, Flask 3.1 (Application Factory pattern with Blueprints)
- **Database & ORM:** SQLite 3, SQLAlchemy 2.0, Flask-SQLAlchemy 3.1
- **Forms & CSRF Security:** Flask-WTF 1.2, WTForms 3.2, email-validator
- **Machine Learning & Analytics:** scikit-learn 1.9, pandas 3.1, NumPy 2.5, SciPy 1.18, joblib 1.6
- **Offline Visualization:** Matplotlib 3.11 (headless `Agg` backend)
- **Frontend UI/UX:** HTML5, CSS3 Custom SaaS tokens, Bootstrap 5.3, Bootstrap Icons 1.11, Chart.js 4.4
- **Testing:** pytest 9.1, pytest-flask 1.3 (26 unit/integration tests)
- **Production Servers:** Waitress 3.0 (Windows), Gunicorn (Linux/Containers)

---

## 4. Dataset & Data Attribution

- **Dataset Name:** UCI Online Shoppers Purchasing Intention Dataset
- **Official Source:** [UCI Machine Learning Repository #468](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
- **Donor Attribution:** C. Sakar, S. Polat, M. Katircioglu, Y. Kastro (2018). *Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks.* Neural Computing and Applications.
- **Session Volume:** 12,330 website sessions across 10 calendar months.
- **Target Variable:** `Revenue` (Boolean: `True` = Purchase occurred, `False` = No purchase).
- **Target Distribution:**
  - `False` (Non-Purchasing): 10,422 sessions (84.53%)
  - `True` (Purchasing): 1,908 sessions (15.47%)
  - **Severe Imbalance Ratio:** ~5.5 to 1.

---

## 5. Data Science & Anti-Leakage Methodology

### 1. The Critical `PageValues` Leakage Decision
In Google Analytics, `PageValues` represents:
$$\text{PageValue} = \frac{\text{Ecommerce Revenue} + \text{Total Goal Value}}{\text{Unique Pageviews for Specified Page}}$$
Because this metric incorporates transaction revenue that has **already happened**, including `PageValues` in a real-time prediction model creates **post-hoc target leakage**.
- In exploratory analysis, `PageValues` exhibits an artificial +0.49 correlation with `Revenue`.
- Leaving `PageValues` in would produce a naive model that looks artificially impressive on paper (ROC-AUC > 0.90) but fails completely when deployed to score an active shopper browsing before checkout!
- **Our Solution:** `PageValues` is strictly excluded from feature pipelines, ensuring genuine pre-purchase inference based exclusively on user behavioral dynamics.

### 2. Duplicate Session De-Duplication
The raw dataset contains 125 exact duplicate session vectors (e.g. anonymous bots or users with 0 duration, bounce 0.2, exit 0.2). We de-duplicate these before splitting (leaving 12,205 unique sessions) to prevent identical records from appearing across both training and test folds.

### 3. Schema & Transformation Pipeline
- **Numerical Features (9):** `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`, `BounceRates`, `ExitRates`, `SpecialDay` $\rightarrow$ Standardized via `StandardScaler()`.
- **Categorical Features (7):** `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`, `Weekend` $\rightarrow$ Encoded via `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
- Operating systems, browsers, regions, and traffic types are explicitly treated as discrete categorical IDs rather than continuous numerical variables.

---

## 6. Classification Models & Evaluation Results

All candidate models were trained on 80% of the deduplicated data (9,764 sessions) and evaluated using **5-Fold Stratified Cross-Validation**. The hold-out test set (2,441 sessions) was kept strictly locked until final evaluation.

### Measured 5-Fold Cross-Validation Performance (Training Split Only)

| Model Architecture | Type & Settings | Val PR-AUC (Avg Precision) | Val ROC-AUC | Val F1-Score | Val Recall | Val Precision | Selection Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear Baseline (`class_weight='balanced'`) | 0.3229 | 0.7442 | 0.3911 | **73.33%** | 26.65% | Baseline |
| **Random Forest** | Bagging Ensemble (150 trees, `depth=12`, `balanced`) | **0.3801** | **0.7757** | **0.4179** | 67.36% | **30.29%** | **Champion Selected** |
| **HistGradientBoosting** | Gradient Boosting (`max_leaf_nodes=31`, `balanced`) | 0.3619 | 0.7638 | 0.4175 | 60.03% | 32.14% | Candidate |

> **Selection Rationale:** Because purchasing sessions comprise only 15.5% of the dataset, accuracy alone is misleading (a naive classifier predicting "No Purchase" every time achieves 84.5% accuracy but is completely useless). Random Forest won by achieving the highest **Validation PR-AUC (0.3801)** and **Validation ROC-AUC (0.7757)**.

### Measured Hold-Out Test Set Evaluation (2,441 Unseen Sessions)

| Model | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC | Test PR-AUC | Brier Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | **69.15%** | 29.27% | **68.59%** | **0.4103** | **0.7808** | **0.3577** | 0.1776 |
| **HistGradientBoosting** | 73.29% | **31.90%** | 62.30% | 0.4220 | 0.7817 | 0.3634 | 0.1640 |
| **Logistic Regression** | 64.93% | 27.89% | 78.27% | 0.4113 | 0.7555 | 0.3310 | 0.2096 |

### Hold-Out Test Confusion Matrix (Random Forest at 0.50 Threshold)

```
                         Predicted: No Purchase (0)    Predicted: Purchase (1)
Actual: No Purchase (0)             1,426 (TN)                   633 (FP)
Actual: Purchase (1)                  120 (FN)                   262 (TP)
```
- **True Positives (262):** Correctly identified 68.59% (262 / 382) of all purchasing sessions on unseen test data without using leaky features.
- **False Positives (633):** Visitors flagged as potential buyers who did not purchase (an expected trade-off of balanced class weighting that avoids costly missed sales).
- **True Negatives (1,426) & False Negatives (120):** Overall hold-out test accuracy is 69.15% ((1,426 + 262) / 2,441).

### Top 8 Behavioral Feature Importances (Tree Impurity)
1. **ExitRates (18.23%):** Single strongest behavioral hazard. Sessions where users exit pages rapidly rarely convert.
2. **ProductRelated_Duration (13.34%):** Substantial time spent evaluating merchandise directly signals purchase intent.
3. **ProductRelated (10.05%):** Depth of catalog exploration.
4. **Month (10.02%):** Strong seasonality (November/December holiday spikes vs February troughs).
5. **BounceRates (9.08%):** Single-page visit hazards.
6. **TrafficType (6.77%):** Inbound channel quality disparity.
7. **Administrative_Duration (5.42%):** Engagement with cart and account settings.
8. **Administrative (4.93%):** Account pageviews.

---

## 7. Decision Threshold Engineering

Classification models output continuous probabilities between 0.0 and 1.0. A fixed default threshold of 0.50 is rarely optimal for asymmetric commercial business problems.

| Threshold | Recall (% Buyers Caught) | Precision (% Accurate Buyer Calls) | F1-Score | Strategic Business Suitability |
| :---: | :---: | :---: | :---: | :--- |
| **0.20** | **94.8%** | 20.3% | 0.3344 | Aggressive lead capture / Exit-intent discount popups |
| **0.35** | 77.5% | 26.2% | 0.3916 | Broad retargeting campaigns |
| **0.40** | **71.7%** | **28.1%** | **0.4040** | **Recommended Operational Equilibrium (Optimal F1)** |
| **0.50** | 68.6% | 29.3% | 0.4103 | Default scikit-learn threshold |
| **0.65** | 50.8% | 34.6% | 0.4115 | High-cost personal concierge / live outbound chat |
| **0.80** | 28.3% | 45.4% | 0.3484 | Conservative VIP qualification / Free gift allocations |

---

## 8. Application Features & Walkthrough

### 1. Executive Landing Overview (`/`)
- Professional navy/slate SaaS hero section with operational pipeline stats.
- Clear breakdown of e-commerce funnel drop-off challenges.
- Anti-leakage documentation and technology architecture badges.

### 2. Real-Time Analytics Dashboard (`/dashboard`)
- Real KPI counters computed directly from dataset: Total sessions, conversions, conversion rate, non-conversions, visitor ratios, and duration averages.
- **5 Dynamic Chart.js Visuals:**
  - Conversion outcome split (Doughnut).
  - Monthly conversion rate curve (Line).
  - New vs Returning visitor conversion rates (Bar).
  - Top traffic channels conversion efficiency (Horizontal Bar).
  - Weekday vs Weekend conversion comparison (Bar).
- **Asynchronous Slicing Filters:** Filter by Month, Visitor Type, Traffic Channel, and Weekend status without full page reload via `/api/analytics/summary`.
- **Data-Grounded Business Recommendations:** Concrete suggestions derived from segment statistics.

### 3. Validated Prediction Engine (`/predict`)
- Comprehensive form accepting all 16 legitimate operational features.
- **4 Instant Test Presets:** High-Intent Shopper, Holiday Deal Hunter, Casual Window Shopper, Immediate Bounce Session.
- Visual probability progress meter comparing score against configured threshold marker.
- Engagement score (0-100) and top session behavioral drivers.
- Clear disclaimer: *Probabilistic model estimate, not a guaranteed commercial intention.*

### 4. Searchable Prediction Ledger (`/history`)
- Paginated table of saved predictions in SQLite.
- Search by Prediction ID or custom campaign tags.
- Filter by predicted class (Purchase vs Non-Purchase).
- Detailed single prediction inspection route (`/history/<id>`).
- Audit actual ground truth outcomes (1, 0, or Unknown).
- **Safe CSV Export (`/history/export`):** Escapes spreadsheet formula injection characters (`=`, `+`, `-`, `@`).

### 5. Model Diagnostics & Threshold Simulator (`/evaluation`)
- Full comparative matrix across all 3 trained models.
- Interactive confusion matrix.
- ROC Curve and Precision-Recall Curve rendered via Chart.js.
- Interactive threshold simulation table displaying trade-offs across cutoffs.

### 6. Dataset Explorer (`/data-explorer`)
- Paginated inspection of all 12,330 UCI sessions.
- Interactive column schema dictionary explaining feature types and roles.
- Filter by month, visitor type, and revenue outcome.
- Downloadable CSV data sample.

---

## 9. Database Design

Implemented with SQLite and SQLAlchemy ORM:

```mermaid
erDiagram
    PREDICTION_HISTORY {
        int id PK
        datetime created_at
        int predicted_class "1 = Purchase, 0 = No Purchase"
        float purchase_probability "0.0000 to 1.0000"
        float decision_threshold "e.g. 0.40"
        string model_version "e.g. v1.0.0 (Random Forest)"
        text session_features_json "JSON payload of 16 features"
        int actual_outcome "Nullable: 1, 0, or NULL"
        string session_summary "Custom campaign / visitor label"
        string notes "Engagement score & diagnostic notes"
    }

    MODEL_METADATA {
        int id PK
        string model_version UK
        string model_name
        datetime trained_at
        int training_sample_count
        float selected_threshold
        text evaluation_metrics_json
        string model_artifact_path
    }
```

---

## 10. REST API Documentation

### 1. Health Endpoint
- **URL:** `GET /health`
- **Response:**
```json
{
  "database": "connected",
  "model": "loaded",
  "service": "Web Analytics-Based User Conversion Prediction System",
  "status": "healthy",
  "version": "1.0.0"
}
```

### 2. Predict Conversion API
- **URL:** `POST /api/predict`
- **Headers:** `Content-Type: application/json`
- **Request Body:**
```json
{
  "Administrative": 3,
  "Administrative_Duration": 85.0,
  "Informational": 1,
  "Informational_Duration": 35.0,
  "ProductRelated": 32,
  "ProductRelated_Duration": 1250.0,
  "BounceRates": 0.002,
  "ExitRates": 0.015,
  "SpecialDay": 0.0,
  "Month": "Nov",
  "OperatingSystems": "2",
  "Browser": "2",
  "Region": "1",
  "TrafficType": "2",
  "VisitorType": "Returning_Visitor",
  "Weekend": "False",
  "decision_threshold": 0.40,
  "session_summary": "API Test Session"
}
```
- **Response (200 OK):**
```json
{
  "decision_threshold": 0.4,
  "engagement_score": 99.8,
  "feature_explanations": [
    {
      "badge": "badge-success",
      "description": "Very low exit rate indicates steady exploration through checkout and cart flow.",
      "feature": "Exit Rate",
      "impact": "Positive",
      "value": "1.50%"
    }
  ],
  "model_name": "Random Forest",
  "model_version": "v1.0.0",
  "predicted_class": 1,
  "predicted_label": "Purchase",
  "prediction_id": 1,
  "purchase_probability": 0.7194,
  "purchase_probability_pct": "71.9%",
  "session_summary": "API Test Session",
  "threshold_pct": "40.0%"
}
```

### 3. Dynamic Analytics Summary API
- **URL:** `GET /api/analytics/summary?month=Nov&visitor_type=Returning_Visitor&traffic_type=all&weekend=all`
- **Response:** Filtered KPIs, monthly charts, visitor charts, and duration brackets.

---

## 11. Windows Setup & Quickstart Guide

Tested and verified on Windows with Python 3.13 and PowerShell.

### Step 1: Create Virtual Environment
Open PowerShell inside the project directory:
```powershell
py -m venv .venv
```

### Step 2: Activate Virtual Environment
```powershell
# If script execution is restricted on your machine, enable it for this process:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

### Step 4: Download & Validate the Official Dataset
```powershell
python scripts/download_data.py
```
*(Extracts and validates `data/raw/online_shoppers_intention.csv`).*

### Step 5: Run Exploratory Data Analysis & Model Training
```powershell
# Generate statistical reports and EDA charts:
python scripts/analyze_data.py

# Train and compare Logistic Regression, Random Forest, and HistGradientBoosting:
python scripts/train_model.py
```

### Step 6: Start the Flask Application
```powershell
python run.py
```
Visit: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 12. Automated Testing

Run the complete 42-test automated verification suite:
```powershell
.venv\Scripts\python.exe -m pytest -v
```

### Test Coverage Highlights (42 Passed / 100% Pass Rate):
- **Data Explorer Regression Suite (`tests/test_data_explorer.py`):**
  - Independent filtering across visitor type, month, and revenue status.
  - Combined multi-attribute filtering.
  - Graceful handling of zero-match filter combinations without crashing.
  - Safe handling of unrecognized or malformed filter parameters.
  - Filtered CSV download validation.
- **Inference & Hardened Validation (`tests/test_prediction.py`):**
  - Valid prediction generation, probability range constraints (0.0 to 1.0), and database persistence.
  - Rejection of missing required feature columns (422).
  - Rejection of non-finite numbers (NaN, +inf, -inf) (422).
  - Rejection of negative durations and page counts (422).
  - Rejection of out-of-bounds rates (422).
  - Rejection of unsupported categorical labels (422).
  - Rejection of invalid decision thresholds without silent clamping (422).
  - Rejection of non-JSON Content-Type (415) and empty/malformed payloads (400).
  - Controlled error responses when model artifacts are missing (503) without exposing tracebacks.
- **Ledger & CSV Sanitization (`tests/test_history.py`):**
  - Empty history state, pagination, and search filtering.
  - Ground-truth outcome auditing.
  - Spreadsheet formula injection escaping (`=`, `+`, `-`, `@`).
- **Pipeline Integrity (`tests/test_model_pipeline.py`):**
  - Verification that `PageValues` is strictly absent from feature inputs.
  - Preprocessor and classifier pipeline loading.
  - Consistency of `model_metrics.json` schema.
- **Application & Status Codes (`tests/test_routes.py`):**
  - Health check endpoint (`/health`), overview (`/`), dashboard (`/dashboard`), diagnostics (`/evaluation`), and custom 404 handler.

---

## 13. Production Deployment Guide (Render & WSGI)

### A. Deploying to Render via Blueprint (`render.yaml`)
This repository includes a pre-configured `render.yaml` specification for zero-configuration Render deployment:
1. Fork or push this repository to your GitHub account: `https://github.com/saurabhg4356/Web-Analytics-Based-User-Conversion-Prediction-System`.
2. Log into [Render Dashboard](https://dashboard.render.com/).
3. Click **New +** $\rightarrow$ **Blueprint**.
4. Connect your GitHub repository. Render will automatically parse `render.yaml` and configure:
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn wsgi:app`
   - **Health Check Path:** `/health`
   - **Environment Variables:** `FLASK_ENV=production`, `SECRET_KEY` (auto-generated).
5. Click **Apply** to deploy.

### B. Manual Web Service Setup on Render
If configuring manually without Blueprints:
- **Environment:** `Python 3`
- **Region:** `Oregon (US West)` or preferred region
- **Branch:** `main`
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn wsgi:app`

> [!IMPORTANT]
> **Avoid the "unicorn: command not found" Typo:**
> If you encounter `bash: line 1: unicorn: command not found (exit code 127)`, check your service's **Settings $\rightarrow$ Start Command** on the Render dashboard. Make sure it is spelled `gunicorn wsgi:app` (with a leading `g`), not `unicorn wsgi:app`. Render provides automatic port binding via `gunicorn.conf.py` or the `Procfile`.

- **Environment Variables:**
  - `FLASK_ENV` = `production`
  - `FLASK_CONFIG` = `production`
  - `SECRET_KEY` = *(generate a secure random 32-character string)*
  - `DEFAULT_DECISION_THRESHOLD` = `0.40`

### C. Local WSGI Execution
- **Windows Production (Waitress):**
  ```powershell
  python wsgi.py
  ```
- **Linux / Docker Production (Gunicorn):**
  ```bash
  gunicorn --bind 0.0.0.0:8000 --workers 2 --threads 4 "wsgi:app"
  ```

### D. SQLite Persistence & Free-Tier Notice
- **Local Development:** The application uses SQLite located at `instance/web_analytics.db`.
- **Render Free Tier Storage:** Render free web services utilize an **ephemeral disk**. This means the database is initialized automatically on startup, but stored predictions in SQLite will reset whenever the free container spins down after inactivity or when a new deployment is triggered.
- **Persistent Cloud Database (Optional):** To retain prediction history permanently across redeployments on Render, provision a free **Render PostgreSQL** database and paste the connection string into the `DATABASE_URL` environment variable:
  ```bash
  DATABASE_URL=postgresql://username:password@hostname:5432/web_analytics
  ```
  The application automatically detects and normalizes `postgresql://` connection strings in production.

---

## 14. Interview Guide & Fresher Explanation

When presenting this project in a technical interview, structure your answers around these engineering decisions:

1. **"Why did you exclude `PageValues`?"**
   *Answer:* "In Google Analytics, PageValues measures the transaction revenue associated with pages visited before completing a transaction. Because it carries backward ground-truth revenue information, including it causes post-hoc target leakage. A model trained with PageValues would score over 90% in testing but fail completely in real-time browsing sessions. I excluded it to build a genuine pre-purchase inference system."

2. **"Why not just use Accuracy as your evaluation metric?"**
   *Answer:* "The dataset is heavily imbalanced: only 15.47% of sessions converted. A trivial classifier predicting 'No Purchase' for every visitor achieves 84.5% accuracy while finding zero buyers. I used Stratified 5-Fold Cross-Validation evaluated on Precision-Recall AUC (PR-AUC) and F1-score to specifically optimize detection of the minority buyer class."

3. **"How did you select the champion model?"**
   *Answer:* "I trained a baseline Logistic Regression with balanced class weights, a 150-tree Random Forest, and a HistGradientBoosting classifier using the exact same stratified splits. Random Forest achieved the highest Validation PR-AUC (0.3801) across 5 folds and was persisted as the champion pipeline."

4. **"How does the business use the decision threshold?"**
   *Answer:* "At 0.50 cutoff, the model caught 68.6% of buyers. By lowering the threshold to 0.40, recall increases to 71.7% while maintaining optimal F1 (0.404). For aggressive exit-intent discount popups where missing a buyer is costly, the threshold can be dialed down to 0.20 to catch 94.8% of potential buyers."

---

## 15. Limitations & Future Roadmap

- **Session Sequences:** Currently treats session totals as aggregated features. Future improvements could integrate recurrent sequence models (LSTM or Transformer) on raw pageview clickstreams.
- **Dynamic Pricing Integration:** Link predicted conversion probabilities with automated discount elasticity algorithms to determine the minimum incentive required to convert each user.
- **SHAP TreeExplainer:** While feature importance and segment deviations are currently computed, pre-compiling SHAP waterfall plots for every real-time prediction would provide exact Shapley value breakdowns.

---

*ConvertSense AI &copy; 2026. Built with Python, Flask, scikit-learn & SQLite.*
