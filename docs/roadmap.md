# Roadmap

**Revised 2026-09-09.** The original 13-week plan (Jul 25 start) never started building; this revision replaces the stale calendar with a Core-objectives-first plan starting from the actual build date. Core objectives (1–4, matching `docs/PROPOSAL.md` §1.1) get a fixed 7-week schedule (Weeks 1–7) below. Stretch objectives (5–6: AWS-deployed FastAPI service, LangChain+Claude drift agent) get planned Weeks 8–12, but those weeks start counting only once Week 7's Core gate actually passes — per the proposal's explicit priority, Core is not shortchanged to start Stretch work early. 12 weeks total, not 13; the extra rigor in Weeks 1–7 (balance diagnostics, multi-candidate cohort comparison, reproducibility pass) replaced a week the original plan didn't budget for.

## Goal

Build a production-grade system that finds business-changing insights in telecom churn data, validates them with rigorous experimentation, and deploys an ML pipeline with agentic AI monitoring — a story usable in both DS and MLE interviews.

- **DS narrative:** Hypothesis to test in Week 1 EDA — billing errors in a subscriber's first 90 days are a *candidate* churn driver, one of several ranked by effect size, not a foregone conclusion. Published churn-driver research typically finds single-factor risk multipliers in the 1.5x–3x range; anything found here gets stated as an EDA finding, then only claimed as causal after Week 3's propensity-matching analysis, never before. Once validated, a CUPED-based simulated experiment (Week 6) tests a retention intervention against it, and an LLM agent (Weeks 10–11) automates monitoring and stakeholder reporting.
- **ML narrative:** XGBoost churn model, target AUC-ROC 0.75–0.80 on Cell2Cell (a known hard, noisy benchmark dataset — published results cluster around 0.70–0.78) with a cross-dataset validation check on IBM Telco (cleaner, smaller — realistic target 0.84–0.87). Deployed to AWS (S3 + ECR + ECS Fargate/App Runner, Weeks 8–9) behind a FastAPI service (<100ms), tracked in MLflow — plus an agentic monitoring system (Weeks 10–11) that detects drift, diagnoses root cause via an LLM decision tree, and can trigger retraining autonomously, with every decision logged and auditable.

## Datasets

- **Cell2Cell** (71K subscribers, 58 features, Duke/Teradata) — primary training data
- **IBM Telco** (7K subscribers, 21 features) — validation dataset

## Core build — Objectives 1–4 (fixed schedule)

Cross-references `docs/PROPOSAL.md` §1.1. Each week ends with a stated verification checkpoint; a week doesn't get marked done without it.

### Week 1 (Sep 9–15) — Setup + Driver-Agnostic EDA

Env setup, download Cell2Cell (train+holdout) + IBM Telco, verify real schema against proposal assumptions (including whether a billing-error-like field actually exists), full EDA, rank **all 58 features** by effect size vs. churn — not just the billing hypothesis.

- Data dictionary for all 58 Cell2Cell + 21 IBM Telco features, committed
- Ranked driver-candidate table (effect size, not p-value) with real numbers
- **Checkpoint**: table exists and is committed before Week 2 starts

### Week 2 (Sep 16–22) — Cohort Analysis + Driver Selection

Cohort comparison on the top 3–5 candidates from Week 1 (churn-rate lift, CI, cohort sample size each), select one leading driver.

- **Checkpoint**: driver selection is written down with the *reason* (effect size + plausibility + adequate cohort N), not just the name — and it's whatever the data actually showed, not a foregone conclusion

### Week 3 (Sep 23–29) — Causal Validation

Confirm panel-vs-cross-sectional data structure first; propensity score matching (primary — see PROPOSAL.md §1.1 Objective 2) with covariate balance diagnostics; DiD only if data structure actually supports it.

- **Checkpoint**: covariate balance diagnostics pass; ATT estimate exists with a confidence interval, not a bare point estimate

### Week 4 (Sep 30–Oct 6) — Feature Engineering + Baseline Model

Feature engineering pipeline built from what EDA actually surfaced; Cell2Cell provider train/holdout split as final test set (§3.5); SMOTE on training fold only; logistic regression baseline vs. persistence baseline; MLflow tracking begins.

- **Checkpoint**: first real MLflow run logged; baseline beats persistence, with the margin stated

### Week 5 (Oct 7–13) — XGBoost + SHAP

XGBoost + Optuna tuning, MLflow-tracked; SHAP (global + per-prediction); significance test (e.g. DeLong's) vs. baseline; IBM Telco cross-check scoped to whatever driver Week 2 selected (§3.6).

- **Checkpoint**: AUC-ROC lands in the realistic 0.75–0.80 range on Cell2Cell — a 90%+ result means stop and find the leakage, not celebrate

### Week 6 (Oct 14–20) — CUPED-Based Simulated A/B Experiment

Explicit simulation protocol (§1.1 Objective 4: synthetic effect size grounded in published retention-campaign ranges, not the model's own prediction); CUPED implementation; variance-reduction measurement; significance testing on raw vs. CUPED-adjusted outcomes.

- **Checkpoint**: simulation assumptions are written down explicitly as assumptions, not disguised as measured effects

### Week 7 (Oct 21–27) — Reproducibility & Verification Pass

Rerun every notebook top-to-bottom; confirm every number in any summary doc traces to committed code output; write `docs/core_results.md` with real numbers only, "TBD" for anything not solid.

- **Checkpoint**: this is the actual gate before Stretch objectives start — not a formality

## Stretch phases — Objectives 5–6

Weeks below are planned, not just "TBD" — but they start counting from whenever Week 7's Core gate actually passes, not from a fixed calendar date. If Core runs long, everything here shifts by the same amount; these weeks don't compress to protect the Oct 23 date, because that date is already retired (see PROPOSAL.md header).

### Week 8–9 — AWS Deployment (Objective 5)

FastAPI service, containerized and deployed to AWS — this is the objective that turns "AWS" from an unbacked resume line into a demonstrated deployment.

- MLflow-registered model artifact pushed to S3
- Dockerfile for the FastAPI service; image pushed to ECR
- Service deployed to ECS Fargate (or App Runner, decided at start of this week based on actual setup overhead) with a scoped, least-privilege IAM role (PROPOSAL.md §10.5)
- `POST /predict`, `POST /batch-predict`, `GET /health`, `GET /model-info`; target <100ms per prediction
- GitHub Actions: build → push to ECR → deploy pipeline (extends the existing lint/test workflow, doesn't replace it)
- Streamlit dashboard (cohort analysis, A/B results, model performance, drift status) — local/Docker Compose to start
- Access restricted (API key or IP allowlist), not left open to the public internet by default

### Week 10–11 — LLM Drift Monitor + AI Agent (Objective 6)

Agentic data drift monitoring with LLM-powered root cause diagnosis and autonomous retrain decisions.

- Evidently AI drift detection on Cell2Cell features; drift reports written to S3 alongside MLflow artifacts
- LLM agent following a structured diagnosis decision tree
- 3 diagnosis categories: seasonal / product change / pipeline bug
- Autonomous retrain-vs-flag decision with reasoning logged (S3 + CloudWatch Logs)
- CloudWatch alarm on the drift-summary metric — the AWS-native alerting path (PROPOSAL.md §10.3)
- AI explainer: SHAP values → plain-English churn score explanation
- Incident report auto-generator for stakeholders

### Week 12 — Polish + GitHub + Demo + Interview Prep

- README with architecture diagram, setup instructions, business impact — written only once real numbers exist from Week 7 and a real deployment exists from Weeks 8–11
- Demo GIF/video walkthrough (2-3 min) showing the live AWS-deployed endpoint, not just local
- Architecture diagram (incl. the AWS deployment path)
- All notebooks cleaned and commented
- DS and MLE interview stories finalized
- LinkedIn post published

## Tech stack

| Layer | Tools |
|---|---|
| Data | Python 3.11, Pandas, NumPy, DuckDB, Matplotlib, Seaborn |
| DS Analysis | SciPy, Statsmodels, CUPED (custom), SHAP |
| ML | Scikit-learn, XGBoost, Imbalanced-learn (SMOTE), MLflow |
| AI | LangChain, Claude API, ChromaDB/FAISS, Evidently AI |
| Deployment | FastAPI, Docker, Streamlit, GitHub Actions |
| Cloud | AWS — S3, ECR, ECS Fargate/App Runner, CloudWatch, IAM |
| Dev Tools | VS Code + Jupyter, Git/GitHub, pip + venv |

## System architecture (component overview)

Data flow: `Raw CSV → Feature Engineering → XGBoost Model → S3/ECR → ECS Fargate (FastAPI) → Streamlit`, with a parallel path `Drift Monitor → LLM Agent → S3/CloudWatch → Incident Report`. All experiments and models tracked in MLflow.

| Component | Description |
|---|---|
| Data Ingestion Layer | Loads Cell2Cell + IBM Telco, validates on load, DuckDB for SQL-style analysis |
| Feature Engineering Pipeline | Validated driver feature (selected in Week 2, not presumed) + derived features, sklearn Pipeline, pickled for serving |
| Cohort Analysis Engine | Core DS insight module — cohort churn comparison on top candidates, propensity matching, CUPED |
| A/B Experimentation Framework | CUPED-based simulator, sample size calc, SRM detection, significance testing |
| XGBoost Model | Primary classifier, MLflow-tracked, tuned, calibrated, registered |
| SHAP Explainability Module | Per-prediction SHAP values, waterfall plots, global importance |
| Evidently AI Drift Monitor | Feature distribution monitoring vs. training baseline, custom thresholds |
| LLM Drift Diagnosis Agent | LangChain agent; decision tree over seasonal/product-change/pipeline-bug/real-drift; logged reasoning |
| AI Churn Explainer | SHAP values → LLM → plain-English per-customer risk explanation |
| Incident Report Generator | Drift event → LLM → structured stakeholder report |
| FastAPI Serving Layer | `/predict`, `/batch-predict`, `/health`, `/model-info`, <100ms, deployed on ECS Fargate/App Runner |
| AWS Deployment Layer | S3 (model artifacts, drift reports), ECR (image registry), ECS Fargate/App Runner (compute), CloudWatch (logs, alarms), IAM (least-privilege service role) |
| Streamlit Dashboard | Cohort analysis, A/B results, model performance + SHAP, drift status |
| MLflow Tracking | Experiment log, model registry, artifact storage |
| Docker Compose Stack | Local dev only — `docker-compose up` starts API (:8000), Streamlit (:8501), MLflow (:5000) |
