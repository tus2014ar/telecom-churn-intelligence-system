# Roadmap

**Revised 2026-10-04.** One-month plan: **Oct 5 – Nov 4, 2026**, at an assumed **20 hours/week** (about 85 hours total). The earlier 12-week plan (about 7 hours/week) never got past setup; this revision keeps every verification checkpoint and cuts breadth instead: Objective 6 is scoped down to a "lite" version, and the Core reproducibility pass shrinks from a week to about two days. If hours per week drop below 20, Objective 6 is dropped first.

Phases map to `docs/PROPOSAL.md` §1.1. A phase is not done until its gate is met.

## Goal

Build a system that finds business-changing insights in telecom churn data, validates them with rigorous experimentation, and deploys an ML pipeline to AWS with monitoring — a story usable in both DS and MLE interviews.

- **DS narrative:** Hypothesis to test in Phase 1 — billing errors in a subscriber's first 90 days are a *candidate* churn driver, one of several ranked by effect size, not a foregone conclusion. Published churn-driver research typically finds single-factor risk multipliers in the 1.5x–3x range; anything found here is stated as an EDA finding, and only claimed as causal after Phase 2's propensity-matching analysis. Once validated, a CUPED-based simulated experiment (Phase 3) tests a retention intervention against it.
- **ML narrative:** XGBoost churn model, target AUC-ROC 0.75–0.80 on Cell2Cell (a known hard, noisy benchmark dataset — published results cluster around 0.70–0.78), with a cross-dataset check on IBM Telco (cleaner, smaller — realistic target 0.84–0.87). Deployed to AWS behind a FastAPI service (Phase 4), with a lite drift-monitoring step (Phase 5).

## Datasets

- **Cell2Cell** (train 51,047 + holdout 20,000 rows, 58 columns) — primary training data
- **IBM Telco** (7,043 rows, 21 columns) — validation dataset

## Phases

| Phase | Dates | Objectives | Tier |
|---|---|---|---|
| 1. Driver discovery | Oct 5–11 | 1 | Core |
| 2. Causal validation and baseline | Oct 12–18 | 2, start of 3 | Core |
| 3. Modeling and experimentation | Oct 19–25 | 3, 4 | Core |
| 4. Verification and deployment | Oct 26–Nov 1 | Core gate, then 5 | Core gate, Stretch |
| 5. Monitoring-lite and release | Nov 2–4 | 6 (lite), README | Stretch |

### Phase 1 — Driver discovery (Oct 5–11)

- Load and validate both datasets; data dictionary for all 58 Cell2Cell and 21 IBM Telco columns
- EDA: nulls, dtypes, class balance, distributions
- Rank **all 58 features** by effect size vs. churn (point-biserial, Cramér's V, Mann-Whitney U as appropriate), not the billing hypothesis by default
- Cohort analysis on the top 3–5 candidates: churn-rate lift, confidence interval, cohort size
- Select one leading driver, with the reason written down
- **Gate:** ranked candidate table and a written driver-selection rationale are committed

### Phase 2 — Causal validation and baseline (Oct 12–18)

- Confirm panel vs. cross-sectional structure first; propensity score matching (primary), DiD only if the data supports it
- Covariate balance diagnostics; ATT estimate with confidence interval
- Feature engineering pipeline built from what EDA surfaced; Cell2Cell provider holdout as the final test set (PROPOSAL.md §3.5); SMOTE on the training fold only
- Logistic regression baseline vs. persistence baseline; MLflow tracking begins
- **Gate:** balance diagnostics pass; ATT has a CI; the baseline beats persistence by a stated margin, with a real MLflow run logged

### Phase 3 — Modeling and experimentation (Oct 19–25)

This phase packs two objectives into one week and is the main schedule risk. If Objective 3 slips, Objective 6 is cut first.

- **Obj 3:** XGBoost with Optuna tuning (MLflow-tracked), SHAP (global and per-prediction), DeLong test vs. baseline, IBM Telco cross-check scoped to the selected driver (PROPOSAL.md §3.6)
- **Obj 4:** CUPED-based simulated A/B experiment; the synthetic effect size is written down as an assumption grounded in published retention-campaign ranges; variance reduction and significance testing on raw vs. CUPED-adjusted outcomes
- **Gate:** AUC-ROC lands in the realistic 0.75–0.80 range (a 90%+ result means find the leakage, not celebrate); simulation assumptions are labelled as assumptions, not measured effects

### Phase 4 — Verification and deployment (Oct 26–Nov 1)

- **Core verification pass (about 2 days):** rerun every notebook top to bottom; confirm every number in `docs/core_results.md` traces to committed code output; "TBD" for anything not solid
- **Gate before deployment:** Core verification passes. Deployment does not start before it.
- **Obj 5:** MLflow-registered model to S3; Docker image to ECR; FastAPI service on ECS Fargate (or App Runner, chosen at the start based on setup overhead) with a least-privilege IAM role (PROPOSAL.md §10.5); `POST /predict`, `POST /batch-predict`, `GET /health`, `GET /model-info`; GitHub Actions build, push, deploy pipeline that extends the existing lint and test workflow; access restricted by API key or IP allowlist
- **Gate:** the live endpoint responds, with measured latency recorded (target under 100ms; stated as measured only once it is)

### Phase 5 — Monitoring-lite and release (Nov 2–4)

- **Obj 6 (lite):** Evidently AI drift report on Cell2Cell features; a Claude diagnosis step with its reasoning logged; no autonomous retrain, no full decision-tree agent. The fuller agent (seasonal / product-change / pipeline-bug diagnosis, CloudWatch alarms, incident reports) is deferred.
- README: architecture diagram, results, before/after, Skills Demonstrated, written only from numbers that exist in committed code and logs
- **Gate:** every README claim traces to committed code or logs; if Obj 6 or the README would be rushed, drop Obj 6 and keep the README honest

## Deferred (not in the one-month window)

Streamlit dashboard, autonomous retrain-vs-flag decisions, CloudWatch alarms on drift, the SHAP-to-LLM churn explainer, the incident report generator, and Terraform/IaC. These stay in `docs/PROPOSAL.md` as the target design but are not scheduled.

## Tech stack

| Layer | Tools |
|---|---|
| Data | Python 3.11, Pandas, NumPy, DuckDB, Matplotlib, Seaborn |
| DS Analysis | SciPy, Statsmodels, CUPED (custom), SHAP |
| ML | Scikit-learn, XGBoost, Imbalanced-learn (SMOTE), MLflow |
| AI | Claude API, Evidently AI (lite monitoring) |
| Deployment | FastAPI, Docker, GitHub Actions |
| Cloud | AWS — S3, ECR, ECS Fargate/App Runner, IAM |
| Dev Tools | VS Code + Jupyter, Git/GitHub, pip + venv |

## System architecture (component overview)

Data flow: `Raw CSV → Feature Engineering → XGBoost Model → S3/ECR → ECS Fargate (FastAPI)`, with a lite parallel path `Evidently drift report → Claude diagnosis (logged)`. All experiments and models tracked in MLflow.

| Component | Description | Phase |
|---|---|---|
| Data Ingestion Layer | Loads Cell2Cell + IBM Telco, validates on load, DuckDB for SQL-style analysis | 1 |
| Cohort Analysis | Cohort churn comparison on top candidates | 1 |
| Causal Validation | Propensity score matching, balance diagnostics, ATT + CI | 2 |
| Feature Engineering Pipeline | Validated driver feature (selected in Phase 1, not presumed) plus derived features, sklearn Pipeline | 2 |
| XGBoost Model | Primary classifier, MLflow-tracked, tuned, registered | 3 |
| SHAP Explainability | Per-prediction SHAP values, waterfall plots, global importance | 3 |
| A/B Experimentation Framework | CUPED-based simulator, sample size calc, significance testing | 3 |
| FastAPI Serving Layer | `/predict`, `/batch-predict`, `/health`, `/model-info`, deployed on ECS Fargate/App Runner | 4 |
| AWS Deployment Layer | S3 (model artifacts), ECR (image registry), ECS Fargate/App Runner (compute), IAM (least-privilege role) | 4 |
| Drift Monitor (lite) | Evidently AI drift report plus a logged Claude diagnosis | 5 |
| MLflow Tracking | Experiment log, model registry, artifact storage | 2 onward |
