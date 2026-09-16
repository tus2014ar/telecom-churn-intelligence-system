# Telecom Churn Intelligence System

End-to-end DS and ML platform for churn prediction, causal analysis, experimentation, and autonomous monitoring.

## Business Problem

Telecom churn costs providers billions annually. This project analyzes 71,000 subscribers to find the drivers of churn, validates the strongest driver with rigorous causal analysis and A/B experimentation, and ships a production ML pipeline with agentic AI monitoring.

See [`docs/PROPOSAL.md`](docs/PROPOSAL.md) for the full project proposal, including objectives, business impact, technical architecture, and production operations.

> Status: Phase 1 (Setup + EDA) — in progress.

## Datasets

- **Cell2Cell** (71K subscribers, 58 features) — primary training data
- **IBM Telco** (7K subscribers, 21 features) — validation dataset

See [`data/README.md`](data/README.md) for download instructions.

## Tech Stack

| Layer | Tools |
|---|---|
| Data | Python, Pandas, NumPy, DuckDB |
| DS Analysis | SciPy, Statsmodels, CUPED, SHAP |
| ML | Scikit-learn, XGBoost, Imbalanced-learn, MLflow |
| AI | LangChain, Claude API, ChromaDB, Evidently AI |
| Deployment | FastAPI, Docker, Streamlit, GitHub Actions |

## Project Structure

```
data/           raw + processed data (gitignored, see data/README.md)
notebooks/      EDA, feature engineering, modeling, experimentation, AI agent notebooks
src/            reusable pipeline code (data loading, feature engineering, model, agent, explainer)
api/            FastAPI serving layer
dashboard/      Streamlit dashboard
tests/          pytest test suite
docs/           architecture diagram, business impact summary
```

## Quick Start

```bash
git clone https://github.com/tus2014ar/telecom-churn-intelligence-system.git
cd telecom-churn-intelligence-system
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Roadmap

Full 13-week build plan (phases, weekly tasks, tech stack, architecture) is in [`docs/roadmap.md`](docs/roadmap.md). This README will be filled in with key results, architecture diagram, and business impact as each phase completes.
