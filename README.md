# Telecom Churn Intelligence System

End-to-end DS and ML project for churn prediction, causal analysis, experimentation, and (planned) cloud deployment with agentic monitoring.

> **Status: Core build in progress — Week 1 (EDA + driver ranking). No results yet.** Every metric in this repo is "TBD" until it exists in committed code with a reproducible output.

## Business Problem

Telecom churn costs providers billions annually. This project finds the strongest driver of churn across the Cell2Cell dataset, validates it causally (not just by correlation), tests a retention intervention in a CUPED-based simulated experiment, and builds a churn classifier with explainability. Stretch goals deploy it to AWS and add LLM-driven drift monitoring.

Stakeholder framing: a **Telecom Retention Lead** who needs to know which customers are at risk, *why*, and whether an intervention would actually work.

Full details: [`docs/PROPOSAL.md`](docs/PROPOSAL.md).

## Objectives

| # | Objective | Tier | Status |
|---|---|---|---|
| 1 | Identify the strongest churn driver across all 58 Cell2Cell features (effect-size ranking, then cohort analysis) | Core | In progress |
| 2 | Validate it causally (propensity score matching; DiD only if the data has panel structure) | Core | Not started |
| 3 | Tuned XGBoost vs. logistic regression and persistence baselines, with SHAP; target AUC-ROC 0.75–0.80 on Cell2Cell | Core | Not started |
| 4 | CUPED-based *simulated* A/B experiment on the validated driver | Core | Not started |
| 5 | FastAPI service deployed to AWS (S3, ECR, ECS Fargate/App Runner, CloudWatch) | Stretch | Not started |
| 6 | LangChain + Claude drift-diagnosis agent with Evidently AI | Stretch | Not started |

Stretch work starts only after Core is verified complete.

## Methodology notes

- The billing-error hypothesis is a *candidate* driver, not a presumed answer. Cell2Cell has no explicit billing-error column, so any billing signal would be a constructed proxy; the data decides the driver.
- Features are ranked by effect size, not p-value (at 71K rows nearly everything is "significant").
- Final test set is Cell2Cell's provider-defined holdout split; neither dataset has per-row timestamps, so a chronological split isn't available.
- The A/B experiment is simulated: the treatment effect is an explicitly stated assumption, not a measured result.
- An AUC-ROC of 90%+ on Cell2Cell would be treated as a leakage bug to find, not a result.

## Datasets

- **Cell2Cell** (train 51,047 + holdout 20,000 rows, 58 columns) — primary training data
- **IBM Telco** (7,043 rows, 21 columns) — cross-dataset validation

Raw data is gitignored. See [`data/README.md`](data/README.md) for download instructions; place files in `data/raw/cell2cell/` and `data/raw/ibm_telco/`.

## Tech Stack

| Layer | Tools |
|---|---|
| Data | Python, Pandas, NumPy, DuckDB |
| DS Analysis | SciPy, Statsmodels, CUPED (custom), SHAP |
| ML | Scikit-learn, XGBoost, Imbalanced-learn, MLflow |
| AI (Stretch) | LangChain, Claude API, Evidently AI |
| Deployment (Stretch) | FastAPI, Docker, Streamlit, GitHub Actions |
| Cloud (Stretch) | AWS — S3, ECR, ECS Fargate/App Runner, CloudWatch, IAM |

## Project Structure

```
data/           raw + processed data (gitignored, see data/README.md)
notebooks/      EDA, causal validation, modeling, experimentation (planned)
src/            reusable pipeline code (planned)
api/            FastAPI serving layer (Stretch)
dashboard/      Streamlit dashboard (Stretch)
tests/          pytest test suite
docs/           PROPOSAL.md, roadmap.md
```

Only `docs/`, `data/README.md`, and empty package scaffolding exist so far.

## Quick Start

```bash
git clone https://github.com/tus2014ar/telecom-churn-intelligence-system.git
cd telecom-churn-intelligence-system
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Roadmap

7 fixed weeks for Core Objectives 1–4 (through Oct 27), then Weeks 8–12 for Stretch, counted from whenever Core's verification gate passes. Weekly checkpoints and detail: [`docs/roadmap.md`](docs/roadmap.md).

A results section, architecture diagram, and quantified business impact will be added only once real numbers exist.
