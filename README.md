# RetainOps

**Customer Retention Policy & Budget Optimization Simulator**

---

## Overview

RetainOps is an executive decision-support system that combines machine learning-powered churn prediction with budget-constrained policy simulation. The system helps leadership teams optimize customer retention strategies by identifying at-risk customers, quantifying financial exposure, and allocating interventions under budget constraints.

**Business Problem**: Organizations face the challenge of allocating limited retention budgets across a customer base with varying churn probabilities and revenue values. RetainOps formalizes this decision-making process by converting qualitative risk into quantitative metrics and optimizing intervention allocation to maximize expected revenue preservation.

---

## Live Demo

**Deployed Application**: [https://retainops.streamlit.app](https://retainops.streamlit.app)

Upload a CSV file with customer data to test the system instantly. The application automatically detects churn columns, processes features, trains the model, and generates retention recommendations.

---

## Key Objectives

- **Predict churn risk** using probabilistic machine learning models
- **Quantify financial exposure** by calculating revenue-at-risk for each customer
- **Optimize intervention allocation** within budget constraints using risk-value scoring
- **Compare multiple strategies** (conservative, balanced, aggressive) to evaluate trade-offs
- **Generate explainable insights** using rule-based reasoning and feature importance analysis
- **Support executive decision-making** with actionable recommendations and scenario analysis

---

## System Capabilities

- **Churn Prediction**: CatBoost-based probabilistic risk modeling with stratified validation
- **Explainability**: Rule-based interpretation of model outputs using feature importance and business heuristics
- **Retention Policy Simulation**: Budget-constrained allocation engine with configurable risk tolerance
- **Budget Optimization**: Greedy algorithm that prioritizes customers based on risk, value, and expected ROI
- **Scenario Comparison**: Multi-strategy analysis (No Intervention, Conservative, Balanced, Aggressive)
- **Risk Quantification**: Monte Carlo simulation for best/worst case outcome estimation
- **Decision Support**: Executive dashboard with KPIs, visualizations, and actionable insights

---

## High-Level Architecture

```
CSV Upload
    ↓
Data Processing & Validation
    ↓
Feature Preprocessing (Categorical/Numeric Detection)
    ↓
Churn Prediction Model (CatBoost)
    ↓
Explainable ML Reasoning Layer (Rule-Based)
    ↓
Revenue-at-Risk Calculation
    ↓
Retention Policy Simulator
    ↓
Budget Optimization Engine
    ↓
UI Outputs (KPIs, Charts, Recommendations)
```

---

## Architecture Components

### Data Ingestion & Validation
- CSV file upload with automatic schema detection
- Churn column auto-detection (supports multiple naming conventions)
- Data quality validation (minimum rows, column requirements)
- Missing value handling (median for numeric, mode for categorical)
- Duplicate removal and type coercion

### Feature Preprocessing
- Automatic categorical vs. numeric feature identification
- Low-cardinality numeric columns treated as categorical
- Standardized binary encoding for churn target
- Feature separation for model training

### Churn Prediction Model
- **Algorithm**: CatBoost Classifier
- **Configuration**: Conservative hyperparameters (depth=4, L2=10, learning_rate=0.05)
- **Validation**: Stratified 80/20 train/validation split
- **Output**: Probabilistic churn risk scores (0-1) for all customers
- **Performance**: 70-85% accuracy with emphasis on probability calibration

### Explainable ML Reasoning Layer
- Feature importance interpretation with business context
- Rule-based explanation generation using templates and heuristics
- Deterministic insights derived from model outputs and statistics
- No external API dependencies; fully self-contained

### Retention Policy Simulator
- Intervention scoring function combining churn probability, customer value, and ROI
- Configurable risk tolerance parameter (prioritizes risk vs. value)
- Greedy allocation algorithm within budget constraints
- Expected outcome calculation with intervention effectiveness modeling

### Budget Optimization Engine
- Revenue-at-risk calculation (customer revenue × churn probability)
- Intervention priority scoring with ROI consideration
- Budget-constrained allocation (sorts by score, allocates until budget exhausted)
- Scenario generation with different risk tolerance levels

### Streamlit UI Layer
- Executive-style dashboard with KPI-first layout
- Six-tab navigation (Overview, Customer Insights, Churn Risk Model, Retention Policy Simulator, Scenario Comparison, Explainable Insights)
- Interactive visualizations (Plotly charts)
- Sidebar-as-control-plane for business parameters
- Real-time simulation and results display

---

## Explainability & Decision Logic

- **Feature Importance Analysis**: Top predictive features identified and ranked by model importance scores
- **Business Interpretation**: Feature names mapped to business concepts (e.g., "contract_type" → "customer commitment level")
- **Threshold-Based Reasoning**: Statistical thresholds used to categorize risk levels and generate recommendations
- **Deterministic Explanations**: Template-based insights derived from metrics (ROI, budget utilization, efficiency ratios)
- **Scenario Logic**: Keyword-based question analysis for what-if scenarios using rule-based comparison
- **Risk Interpretation**: Monte Carlo statistics (mean, percentiles, coefficient of variation) explained in business terms
- **Actionable Recommendations**: Strategic guidance based on performance metrics and efficiency analysis

---

## Sample Datasets Included

### `sample_customer_data.csv`
- **Size**: 100 customers
- **Purpose**: Quick testing and validation
- **Characteristics**: Representative, privacy-safe dataset with realistic churn patterns
- **Features**: Customer demographics, contract types, payment methods, service types, revenue data
- **Usage**: Upload directly into the app for immediate testing

### `realworld_customer_data.csv`
- **Size**: 300 customers
- **Purpose**: Enterprise-scale testing and demonstration
- **Characteristics**: Larger, more complex dataset mimicking real-world enterprise customer data
- **Features**: Extended feature set with realistic distributions and correlations
- **Usage**: Upload directly into the app to test system performance on larger datasets

Both datasets can be uploaded directly into the application to test all functionality without additional data preparation.

---

## Expected Input Format

- **Customer-level features**: Any combination of numeric and categorical columns
- **Common features include**:
  - Tenure metrics (months, years of relationship)
  - Engagement indicators (usage frequency, feature adoption)
  - Usage metrics (transaction volume, service consumption)
  - Complaint history (support tickets, escalations)
  - Behavioral metrics (login frequency, feature usage)
  - Demographics (age, region, segment)
  - Contract information (type, duration, pricing tier)
  - Payment information (method, history, status)
- **Churn column**: Binary target (0/1, True/False, or column name containing "churn")
- **Optional revenue column**: For accurate revenue-at-risk calculations
- **Flexible schema**: System auto-detects feature types and handles missing values

---

## Project Structure

```
RetainOps/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── EXPLANATION.md                  # Comprehensive system documentation
├── sample_customer_data.csv        # Sample dataset (100 customers)
├── realworld_customer_data.csv     # Real-world dataset (300 customers)
├── .streamlit/
│   └── config.toml                # Streamlit theme configuration
└── src/
    ├── data_utils.py              # Data loading, validation, cleaning
    ├── eda.py                     # Exploratory data analysis, segmentation
    ├── ml_model.py                # CatBoost churn risk modeling
    ├── revenue.py                 # Revenue at risk calculations
    ├── policy_simulator.py        # Budget-constrained allocation logic
    ├── evaluation.py              # Scenario comparison and metrics
    └── llm_helper.py              # Rule-based explainable insights generator
```

---

## Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start application
python -m streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

**Windows Note**: If `streamlit` command is not recognized, use `python -m streamlit run app.py`.

---

## Design Principles

- **Interpretability**: All model outputs and recommendations are explainable using feature importance and business logic
- **Reproducibility**: Fixed random seeds, deterministic algorithms, version-controlled dependencies
- **Modularity**: Clean separation of concerns (data processing, modeling, simulation, UI)
- **Deployment Readiness**: No external API dependencies, self-contained, production-stable
- **Business Alignment**: Metrics and outputs designed for executive decision-making (ROI, revenue-at-risk, net benefit)

---

## Target Roles

- **Machine Learning Engineer**: Production ML pipeline, model training, feature engineering, deployment
- **Data Scientist**: Statistical analysis, model validation, business insights, experimental design
- **AI Engineer**: Explainable AI, rule-based reasoning, decision support systems
- **Product / Strategy**: Business metrics, scenario planning, budget optimization, retention strategy

---

## Summary

- **Decision Support System**: RetainOps transforms customer retention from reactive service into strategic, data-driven resource allocation
- **End-to-End Pipeline**: Complete workflow from raw customer data to actionable retention recommendations with explainable insights
- **Production-Ready**: Deterministic, self-contained system suitable for deployment without external dependencies or API keys
