# Telecom Churn Intelligence System — Project Proposal

**Type**: Independent DS/ML portfolio project
**Author**: Tushar
**Build start**: 9 September 2026
**Core objectives (1–4) target**: 27 October 2026 — fixed 7-week schedule, see `docs/roadmap.md`
**Stretch objectives (5–6) + deployment**: dates TBD, begin only after Core is verified complete; the original 23 October deploy target is no longer achievable given the actual start date and is retired rather than left stale
**Status**: Core build in progress (Week 1)
**Revision**: 9 September 2026 — methodology resolved for Objectives 1–4, schedule replanned from actual start date

---

## Executive Summary

The Telecom Churn Intelligence System is an end-to-end DS and ML platform that finds the strongest driver of subscriber churn in a 71,000-subscriber telecom dataset, validates it with causal analysis and a CUPED-based A/B experiment, and ships a production churn-scoring pipeline with agentic AI monitoring. It is designed around one stakeholder — a **Telecom Retention Lead** — who needs to know not just which customers are likely to churn, but *why*, whether a proposed intervention actually moves the needle, and whether the deployed model can be trusted to keep working without manual babysitting. The system covers the full lifecycle: EDA and cohort analysis, causal validation, a tuned XGBoost classifier with SHAP explainability, a CUPED-adjusted experimentation framework, a FastAPI/Docker/Streamlit production layer, and an LLM-driven drift-diagnosis agent that reasons about *why* the model might be degrading rather than only flagging that it has. It is built to support two distinct interview narratives — data scientist and ML engineer — from a single, real system.

---

## 1. Purpose & Objectives

Telecom churn costs providers billions annually, and most churn-prediction projects stop at a leaderboard AUC score. This project is scoped to go further: find a driver, prove it's causal (not just correlated), test an intervention against it with proper statistical rigor, and deploy the resulting model as a system that monitors and explains itself in production.

### 1.1 Core Objectives

- **Objective 1 (Core)**: Identify the strongest driver(s) of churn across all 58 Cell2Cell features through EDA and cohort analysis. Method: rank every feature by effect size against churn (point-biserial correlation / Cramér's V / Mann-Whitney U as appropriate to type), not by p-value alone — at 71K rows almost everything is "significant," so effect size is what separates a real driver from noise. The top 3–5 candidates get full cohort comparison (churn-rate lift, confidence interval, cohort sample size) before one is selected. The proposal's working hypothesis (a billing-error-related signal) is a candidate to test, not a predetermined answer — if EDA surfaces a stronger, unrelated driver, that's what gets carried into Objective 2.
- **Objective 2 (Core)**: Validate the leading driver causally. Method depends on data structure, confirmed in Week 1: if Cell2Cell is a single cross-sectional snapshot per customer (expected — no repeated per-customer observations over time), propensity score matching is the feasible method, with covariate balance diagnostics (standardized mean differences) reported and an ATT estimate given with a confidence interval, not a point estimate alone. Difference-in-differences is only used if the data turns out to have genuine panel structure; this is checked before either method is committed to.
- **Objective 3 (Core)**: Build and tune an XGBoost churn classifier — target AUC-ROC 0.75–0.80 on Cell2Cell, in line with published benchmarks for this dataset (a known hard, noisy one; claims of 90%+ on it typically indicate leakage) — with SHAP-based explainability, evaluated against both a logistic regression baseline and the mandatory persistence baseline. Evaluation is class-specific: precision/recall/F1 on the churn-positive (minority) class, not overall accuracy alone. Model-vs-baseline comparison uses a significance test (e.g. DeLong's test on AUC), not a bare metric delta.
- **Objective 4 (Core)**: Design a CUPED-based A/B experiment simulating a retention intervention targeted at the validated driver. Since no real intervention outcome exists in historical data, the simulation protocol is stated explicitly rather than left implicit: split a held-out cohort into control/treatment, inject an assumed effect size grounded in published retention-campaign literature (roughly high-single-digit to ~15–20% relative churn reduction — not the 30% figure earlier drafts of this narrative used), then measure CUPED's variance reduction and the significance test result on both raw and CUPED-adjusted outcomes. Every report of this result states plainly that the treatment effect is a simulated assumption, not a measured one.
- **Objective 5 (Stretch)**: Deploy the classifier as a FastAPI service, containerized with Docker, to **AWS** — MLflow-registered model artifacts pushed to S3, image built and pushed to ECR, served from ECS Fargate (or App Runner if Fargate proves more setup than a solo project needs — decided when this objective starts, not now), with a GitHub Actions pipeline that builds, tests, and deploys on push. A Streamlit dashboard (cohort analysis, A/B results, model performance, drift status) runs alongside, initially local/Docker Compose. This is the objective that turns "AWS" from an unbacked line in Technical Skills into a demonstrated deployment.
- **Objective 6 (Stretch)**: Build an agentic monitoring layer (LangChain + Claude) that detects data drift (Evidently AI), diagnoses probable root cause via a structured decision tree, and recommends retrain-vs-flag — with every decision logged and auditable. Drift reports and agent decision logs are written to S3 alongside MLflow artifacts; CloudWatch is the alerting channel once the service is deployed to AWS (§10.4), rather than defaulting straight to Slack/email for a project that already has an AWS presence.

### 1.2 Business Impact

Every objective ties to a decision a Telecom Retention Lead would make differently because of this system:

- A validated (not merely correlated) churn driver changes what the retention team actually spends money fixing, rather than chasing a spurious correlation.
- A CUPED-adjusted experiment read gives a defensible answer to "did the intervention work?" with less noise and a smaller required sample than a naive A/B test — meaning faster, cheaper answers.
- SHAP-based per-customer explanations turn a bare risk score into something a retention rep can act on ("this customer is at risk because of X, Y, Z") rather than a black-box number.
- An agentic drift monitor that distinguishes seasonal shift, product change, pipeline bug, and real concept drift means fewer false-alarm retrains and fewer silently-stale models — the model degrading undetected is the actual production failure mode this guards against.

**Estimated value framing**: if the validated driver enables an intervention that reduces churn by even a few points in the highest-risk cohort, the retained-revenue impact scales directly with subscriber base and average revenue per user; the exact dollar figure depends on assumptions not yet validated against real experiment results, and this proposal deliberately does not state one as fact until Objective 4 produces it.

---

## 2. Business & Research Questions

| # | Type | Question |
|---|------|----------|
| Q1 | Discovery | Which features most strongly separate churned from retained subscribers across the 71K Cell2Cell dataset? |
| Q2 | Causal | Is the leading candidate driver causally linked to churn, or does it merely correlate with some other underlying cause? |
| Q3 | Experimentation | Does a simulated retention intervention targeted at the validated driver produce a statistically significant reduction in churn, using CUPED to reduce variance? |
| Q4 | Prediction | Can a tuned XGBoost classifier beat a logistic regression baseline by a meaningful margin, and does it generalize to the smaller, independently-sourced IBM Telco dataset? |
| Q5 | Production | Can the deployed model detect and correctly diagnose its own drift (seasonal vs. product change vs. pipeline bug vs. real drift) without a human first noticing degraded performance? |

---

## 3. Data Sources

### 3.1 Primary Dataset — Cell2Cell

71,000 subscribers, 58 features, from the Duke/Teradata Center for Customer Relationship Management (public, via Kaggle). This is the primary training and causal-analysis dataset.

### 3.2 Validation Dataset — IBM Telco Customer Churn

~7,000 subscribers, 21 features (public IBM sample dataset, via Kaggle). Used to check whether findings — especially the leading churn driver and the trained model's behavior — generalize beyond the primary dataset, rather than being an artifact of one data source.

### 3.3 Known Gaps

- No real-time behavioral or usage-event stream — both datasets are static snapshots, not a live production feed. The "production" pipeline (§7) scores against this snapshot data; a genuinely live deployment would need a real ingestion source, which is out of scope here.
- No real experiment — the CUPED A/B framework (Objective 4) is built and validated methodologically, but run as a simulation against historical/held-out data, not a live randomized rollout to real subscribers. This distinction is stated explicitly in the dashboard and any interview narrative, not glossed over.

### 3.4 Data Validation Approach

Both datasets are checked at load time for schema (expected columns, types), plausible ranges (no negative tenure, no impossible usage values), and class balance, before entering the feature-engineering pipeline — so a malformed load fails fast rather than silently producing a corrupted feature set.

### 3.5 Train/Test Split Strategy

Neither dataset carries a per-row event timestamp, so a literal chronological split isn't available. Cell2Cell ships with its own provider-defined train/holdout files; those are used as the final, leak-resistant test set (the closest equivalent to a non-random split this dataset structurally supports). Internal train/validation splitting during development uses stratified sampling on the churn label. This is a deliberate substitution for date-ordered splitting, stated here rather than silently reinterpreted.

### 3.6 Cross-Dataset Validation Scope

IBM Telco's schema (21 features) does not mirror Cell2Cell's (58 features) column-for-column, so the Objective 1/Q4 cross-dataset check is necessarily scoped to whatever the actual leading driver turns out to be: it checks whether an equivalent or closely analogous signal exists in IBM Telco and points the same direction, not that an identical column reproduces there. The specific mapping is finalized only after Objective 1 selects a driver.

---

## 4. Technical Architecture

| Layer | Stage | Tools / Approach |
|---|---|---|
| Data | Ingestion | Pandas/DuckDB load of Cell2Cell + IBM Telco; schema validation on load. |
| Data | EDA & Cohort Analysis | Matplotlib/Seaborn distributions; all 58 features ranked by effect size (not p-value) vs. churn; top 3–5 candidates get full cohort comparison (§1.1, Objective 1). |
| DS Analysis | Causal Validation | Propensity score matching (primary, given expected cross-sectional structure) with covariate balance diagnostics and ATT + CI; difference-in-differences only if the data proves to have panel structure (§1.1, Objective 2). |
| DS Analysis | Experimentation | CUPED variance reduction (custom implementation); simulated A/B test with an explicitly stated synthetic effect-size assumption (§1.1, Objective 4); significance testing (SciPy). |
| ML | Feature Engineering | Derived features (tenure buckets, usage trend, driver flag); sklearn Pipeline; SMOTE for class imbalance (train split only). |
| ML | Modeling | Logistic regression baseline → tuned XGBoost; MLflow experiment tracking and model registry. |
| ML | Explainability | SHAP — waterfall, beeswarm, global importance; feeds the AI explainer. |
| AI | Drift Monitoring | Evidently AI drift reports on production-simulated data vs. training baseline. |
| AI | Drift Diagnosis Agent | LangChain + Claude API; structured decision tree (seasonal / product change / pipeline bug / real drift); retrain-vs-flag decision logged. |
| AI | Churn Explainer | SHAP values → Claude API → plain-English per-customer risk explanation. |
| AI | Incident Reporting | Drift event → Claude API → structured stakeholder report. |
| Deployment | Serving | FastAPI: `POST /predict`, `POST /batch-predict`, `GET /health`, `GET /model-info`; target <100ms per prediction. |
| Deployment | Dashboard | Streamlit: cohort analysis, A/B results, model performance + SHAP, drift status. |
| Deployment | Packaging | Docker for the FastAPI image; Docker Compose for local dev (API + Streamlit + MLflow, one command). |
| Deployment | Cloud (AWS) | S3 — model artifacts (from MLflow registry) and drift-report storage. ECR — container image registry. ECS Fargate (or App Runner) — serves the FastAPI container. CloudWatch — logs and drift/accuracy alerting once deployed. |
| Deployment | CI/CD | GitHub Actions — lint/test on every push; build → push to ECR → deploy, gated to run only once Objective 5 starts (not before). |

Full phase-by-phase build sequence is in [`docs/roadmap.md`](roadmap.md).

---

## 5. Repository Structure

Current state: `api/`, `data/`, `src/`, `tests/` exist as empty package scaffolding; `docs/` holds this proposal and the roadmap. No modeling or pipeline code has been written yet. Target structure:

```
telecom-churn-intelligence-system/
├── data/
│   ├── raw/                # gitignored — Cell2Cell + IBM Telco source files
│   └── processed/          # gitignored — cleaned/feature-engineered outputs
├── notebooks/               # 01_eda, 02_feature_engineering, 03_modeling,
│                             # 04_experimentation, 05_ai_agent
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── agent.py             # LangChain drift-diagnosis agent
│   └── explainer.py         # SHAP -> LLM churn explainer
├── api/
│   ├── main.py
│   ├── schemas.py
│   └── dependencies.py
├── dashboard/
│   └── app.py                # Streamlit
├── tests/
│   ├── test_api.py
│   ├── test_model.py
│   └── test_agent.py
├── docs/
│   ├── PROPOSAL.md            # this document
│   ├── roadmap.md
│   ├── architecture.md        # planned
│   └── business_impact.md     # planned
├── .github/workflows/          # CI (lint/test, always) + deploy.yml (build->ECR->ECS, added at Objective 5)
├── deploy/
│   ├── ecs-task-def.json       # ECS task definition (added at Objective 5)
│   └── iam-policy.json         # least-privilege policy for the service role (§10.5)
├── docker-compose.yml           # local dev only
├── Dockerfile                   # built and pushed to ECR at Objective 5
├── requirements.txt
└── README.md
```

---

## 6. Full ML Pipeline Coverage

| Standard ML Lifecycle Stage | Coverage |
|---|---|
| Business understanding | Purpose, objectives, named stakeholder (§1) |
| Data collection | Cell2Cell (primary) + IBM Telco (validation) (§3) |
| Exploratory data analysis | Notebook: distributions, cohort comparison, driver candidate identification |
| Causal validation | Propensity matching / DiD on leading driver (§4) |
| Data cleaning & feature engineering | Missing-value handling, derived features, SMOTE (train only) |
| Model training with baseline | Logistic regression baseline → tuned XGBoost (§4) |
| Evaluation | AUC-ROC/AUC-PR, precision-recall, cross-dataset validation on IBM Telco |
| Experimentation | CUPED-adjusted simulated A/B test with significance testing |
| Explainability | SHAP (global + per-prediction), LLM plain-English layer |
| Deployment | FastAPI + Docker, <100ms target latency |
| Monitoring | Evidently AI drift detection + LLM diagnosis agent |
| Iteration | Agent-recommended retrain-vs-flag decision, logged |
| Reproducibility | MLflow experiment tracking and model registry |
| Serving / interface | FastAPI endpoints + Streamlit dashboard |

---

## 7. DS/ML Concepts Applied

**Core statistics and machine learning**
- Binary classification with class imbalance (SMOTE, applied to training data only)
- Causal inference: propensity score matching / difference-in-differences
- Variance reduction in experimentation: CUPED
- Statistical significance testing on experiment results, not point estimates alone
- Ensemble methods (XGBoost) with hyperparameter tuning and MLflow-tracked experiments
- Model explainability (SHAP), both global and per-prediction
- Cross-dataset generalization check (train on Cell2Cell, validate signal on IBM Telco)

**Applied LLM / agentic engineering**
- Structured decision-tree reasoning for drift diagnosis (not open-ended generation)
- SHAP-to-natural-language explanation generation
- Logged, auditable agent decisions (retrain vs. flag) — every automated decision has a recorded rationale

**MLOps / production engineering**
- Experiment tracking and model registry (MLflow)
- Containerized serving (Docker for the FastAPI image; Docker Compose for local dev)
- Cloud deployment (AWS): S3 for model artifacts, ECR for the image registry, ECS Fargate/App Runner for serving, CloudWatch for logs/alerting, least-privilege IAM scoping (§10.5)
- Data drift monitoring (Evidently AI) as a distinct concern from model retraining
- CI/CD (GitHub Actions) — lint/test gate every push; build-push-deploy pipeline once Objective 5 starts
- Latency-aware API design (<100ms target for single prediction)

---

## 8. Responsible AI & Model Risk

This system informs retention *strategy* and prioritization, not automated action taken directly against a customer's account — outputs are advisory.

- **Human-in-the-loop for LLM output**: auto-generated incident reports and per-customer explanations are outputs a retention team would review before acting on, not auto-executed actions.
- **Explicit uncertainty communication**: the dashboard reports the model's current AUC/precision-recall alongside every prediction batch, not just the prediction itself.
- **No real experiment claims**: because Objective 4's A/B test is simulated (§3.3), all reporting — dashboard, README, and any interview narrative — states this plainly rather than implying a live randomized rollout occurred.
- **Public, non-proprietary data**: both datasets are public Kaggle sources with no real customer PII, so this project carries materially lower data-governance risk than a proprietary-data system, and that distinction is worth naming explicitly rather than assumed.

---

## 9. Model Card (Planned)

A model card will be published (`docs/model_card.md`, planned) once Phase 3 modeling is complete, covering: intended use (churn-risk scoring for retention prioritization, not an automated retention-action trigger), training data (Cell2Cell, with IBM Telco as an out-of-sample check), evaluation results (AUC-ROC/AUC-PR, backtested), and known limitations (static snapshot data, no live usage stream, simulated rather than live experimentation).

---

## 10. Production Operations

### 10.1 Service Expectations

- **Latency target**: <100ms for a single `/predict` call.
- **Availability framing**: this is a portfolio demo service, not a system with a real SLA commitment to paying users — "available" means the deployed ECS/App Runner service responds to `/health` and the dashboard reflects current model state, not a 24/7 uptime guarantee.

### 10.2 Model Promotion

A retrained ("challenger") model is compared against the currently-registered ("champion") model in MLflow before promotion; a challenger that doesn't beat the champion on the primary metric is logged and discarded rather than auto-deployed.

### 10.3 Monitoring & Alerting

The Evidently AI + LLM agent pipeline (§4) produces a structured drift summary and a diagnosis with logged reasoning on every scheduled run; this is surfaced in the Streamlit dashboard's drift-status panel and, once deployed (Objective 6), a CloudWatch alarm on the drift-summary metric — an AWS-native alerting path rather than a Slack/email default chosen mainly because it's easier to wire up.

### 10.4 Incident Response

If a scheduled drift-check or retrain run fails, the last known-good model stays registered and serving; the failure is logged, not silently ignored.

### 10.5 Access Control (AWS)

Once deployed, the ECS/App Runner service and its S3/ECR resources are scoped to a dedicated IAM role with least-privilege permissions (read the model artifact, write drift reports/logs — nothing broader), not a root or admin-level credential. The `/predict` and `/batch-predict` endpoints are not left open to the public internet by default; access is restricted (API key or IP allowlist) since this is a demo service, not a product with real users to serve unauthenticated.

### 10.6 Cost & Resource Considerations

Core (Objectives 1–4) runs entirely on local/free tooling. Once Objective 5 adds AWS, real cost enters the picture and is named explicitly rather than glossed over: ECS Fargate/App Runner, ECR storage, and S3 all have small but non-zero costs at even light portfolio-demo traffic, and the plan is to run the deployed service only while actively demoing it (not left running 24/7 indefinitely) to keep this near the AWS free tier. Claude API calls for the drift-diagnosis agent and explainer remain the other real per-request cost. Stating both costs plainly — not just the existence of a cloud deployment — is itself part of demonstrating a production mindset, not an afterthought.

---

## 11. Roadmap

Full week-by-week build plan (7 fixed weeks for Core Objectives 1–4, then Weeks 8–12 for Stretch Objectives 5–6, counted from whenever Week 7's Core gate actually passes): [`docs/roadmap.md`](roadmap.md).
