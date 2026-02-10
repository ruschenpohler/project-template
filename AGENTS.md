# AGENTS.md

## Project Overview

This is an agent-centric ML experiment platform for regulated banking and
financial consulting. It provides a reproducible, config-driven workflow for
training and evaluating machine learning models with full experiment tracking
via MLflow.

**Design priorities:** Reproducibility, logging/auditability, simplicity.
Every experiment must be fully traceable and re-runnable by anyone, at any time.
If it isn't logged, it didn't happen.

**Stack:** Python 3.11+ | uv | MLflow (SQLite backend) | Pydantic | PyYAML |
pandas | scikit-learn | XGBoost | pytest | pre-commit

**Repository layout:**

```
run_experiment.py          # Single experiment entrypoint
configs/
  experiment.yaml          # Experiment config (what to run)
  reasoning.yaml           # Reasoning metadata (why you're running it)
  project_standards.yaml   # Protected fields, allowed values, taxonomy
src/
  config_schema.py         # Pydantic validation
  data_access.py           # Governed data access layer (CAS-aware)
  features.py              # Feature engineering
  model.py                 # Model training
scripts/                   # Utility scripts (promote_model.py, etc.)
data/
  shadow/
    {sha256_hash}/         # Content-addressed dataset directories
tests/                     # Mirrors src/ structure
logs/
  protected_overrides.log  # Append-only audit trail
```

---

## 1. Entrypoint Rule

**All experiments must be run via:**

```bash
python run_experiment.py --config configs/experiment.yaml
```

No ad hoc training scripts. No calling training, evaluation, or data-processing
functions directly as standalone scripts. `run_experiment.py` is the single
orchestrator that loads config, validates it, sets up MLflow tracking, computes
hashes, and delegates to pipeline stages.

Do not create parallel entrypoints. If you need new pipeline capability, extend
`run_experiment.py` or the modules it calls.

A pre-commit hook enforces this mechanically by scanning for training patterns
(`.fit(`, `.train(`, `.partial_fit(`, `mlflow.start_run`, `mlflow.log_`) in
`src/` files. Exclusions: `run_experiment.py`, `tests/`, `scripts/`, `*.md`.

---

## 2. Config Rules

- Every experiment is defined by **two YAML config files**:
  - `configs/experiment.yaml` -- defines **what** to run (model type,
    hyperparameters, dataset, evaluation settings).
  - `configs/reasoning.yaml` -- defines **why** you are running it (hypothesis
    category, change description, expected effect).
- All config files are validated at load time by Pydantic models in
  `src/config_schema.py`, which reads allowed values from
  `configs/project_standards.yaml`.
- **Never hard-code hyperparameters, paths, or experiment settings** in Python
  source files. All tunable values belong in config YAML.
- Before each run, fill in `configs/reasoning.yaml` with:
  - `hypothesis_category` (must be one of the allowed categories)
  - `change_description` (free text: what you changed and why)
  - `expected_effect` (required for agents: `improve`, `reduce_variance`, or
    `explore`)

### 2.1 Protected Fields

The `protected` block of `configs/experiment.yaml` contains fields that **must
not be changed without explicit human approval**:

| Field | Reference Value | Why It's Protected |
|---|---|---|
| `metric_definition` | `roc_auc` | Changing the primary metric invalidates all historical comparisons |
| `split_strategy` | `stratified_kfold` | Changing the split makes metrics non-comparable across runs |

These reference values are defined in `configs/project_standards.yaml`, which is
the single source of truth for all governance rules.

**Enforcement:**
- `run_experiment.py` compares each protected field against
  `project_standards.yaml` at startup. If any differ and `--override-protected`
  was not passed, the script exits with a clear error listing the mismatched
  fields.
- A pre-commit hook blocks commits where any `configs/experiment*.yaml` file has
  protected field values that differ from `project_standards.yaml`.
- If `--override-protected` is used, an entry is appended to
  `logs/protected_overrides.log` with: timestamp, field name, old value, new
  value, operator, and config file path.

**Before modifying any protected field:** State which field you are changing and
why. Wait for explicit human approval. Use the `--override-protected` flag only
after receiving approval.

### 2.2 Allowed Values (from project_standards.yaml)

| Setting | Allowed Values |
|---|---|
| `allowed_metrics` | `roc_auc`, `f1`, `precision`, `recall`, `log_loss` |
| `allowed_split_strategies` | `stratified_kfold`, `temporal_split` |
| `allowed_model_types` | `xgboost`, `lightgbm`, `logistic_regression`, `random_forest` |
| `hypothesis_categories` | `feature_addition`, `feature_transformation`, `feature_removal`, `hyperparameter_tuning`, `regularization`, `model_architecture`, `data_cleaning`, `data_expansion`, `threshold_tuning`, `ensemble`, `pipeline_refactor` |

---

## 3. Data Rules

### 3.1 Shadow Data Only

- **Only access data through `src/data_access.py`.** This is the governed data
  access layer. Do not read data files directly with pandas or any other method.
- Only use data in `data/shadow/`. Never access or reference production data.
  Production data (if it exists locally) lives in a separate directory excluded
  from agent scope; the data access layer enforces this by only looking in
  `data/shadow/`.

### 3.2 Content-Addressed Storage (CAS)

- Dataset directories are named by the **SHA256 hash** of their contents:
  `data/shadow/{sha256_hash}/`.
- The `dataset_version` field in `configs/experiment.yaml` must equal the
  directory name (the content hash).
- At load time, `src/data_access.py` recomputes the SHA256 of the dataset files
  and verifies it against the directory name. If they don't match, the load
  fails with a clear error.
- The dataset schema hash (column names, types) is also computed and returned
  alongside the data. Both hashes are logged to MLflow as Tier 1 parameters.
- **Datasets are immutable.** Never overwrite or mutate files in a
  content-addressed directory. If data changes, the hash changes, and a new
  directory is created. The old directory remains untouched.

### 3.3 Adding a New Dataset

1. Compute the SHA256 of the dataset files.
2. Create a directory under `data/shadow/` with that hash as the name.
3. Place the dataset files inside.
4. Update `dataset_version` in `configs/experiment.yaml` to the new hash.

---

## 4. File Scope Rules

You may freely modify:
- `src/features.py` -- feature engineering logic
- `src/model.py` -- model training logic
- `configs/experiment.yaml` -- experiment parameters
- `configs/reasoning.yaml` -- reasoning metadata

**Do not modify without explicit human approval:**
- `run_experiment.py` -- the entrypoint is scaffold; changes affect all experiments
- `src/data_access.py` -- the data governance boundary
- `src/config_schema.py` -- the validation layer
- `configs/project_standards.yaml` -- governance rules
- Test files in `tests/` -- tests are the safety net

**One module, one responsibility.** Do not add unrelated functionality to an
existing module. Test files must mirror the module they test
(e.g., `src/model.py` -> `tests/test_model.py`).

---

## 5. Git Rules

- **Work on feature branches.** The main branch is protected. Direct pushes to
  main are blocked. All changes go through pull requests.
- **Include the MLflow run ID** in your commit messages and PR descriptions.
  Every code change that affects modeling must link to an experiment proving what
  happened. This is your audit trail.
- A pre-merge hook (`scripts/verify_pr_run_match.py`) verifies that the MLflow
  run's commit hash matches the branch head commit. If it doesn't match, the
  merge is blocked.
- **Do not commit:**
  - Raw data files or large binaries
  - MLflow artifacts (`mlruns/`, `mlartifacts/`, `mlflow.db`)
  - Virtual environments (`.venv/`, `env/`)
  - Credentials, API keys, `.env` files
- One logical change per commit. Do not bundle unrelated changes.
- Do not force-push to main.

---

## 6. Testing

**Run the full test suite after any change and before committing:**

```bash
uv run pytest tests/
```

All tests must pass. Do not mark a task as done or commit code with failing
tests. Tests serve two purposes: verifying the system works, and giving you a
feedback loop for self-correction.

---

## 7. Modeling Risk Checklist

**This checklist is mandatory.** Before every experiment run, verify each item.
Copy this checklist into your reasoning and complete it. Do not skip items. If
an item is not applicable, write "N/A" with a brief justification.

```
### Pre-Run Risk Checklist

1. [ ] **Missing data:** Have missing values been handled appropriately?
       Are there unexplained drops in feature coverage?

2. [ ] **Data leakage:** Does the feature pipeline use any information that
       would not be available at prediction time? Are there target-correlated
       features that should be excluded?

3. [ ] **Temporal validation:** If the data is time-series, is the train/test
       split temporal? Is the validation strategy appropriate for the data's
       time structure?

4. [ ] **Class imbalance:** Is the target variable imbalanced? Has this been
       accounted for in the model or evaluation metric?

5. [ ] **Protected fields:** Are the protected fields (metric_definition,
       split_strategy) unchanged from project_standards.yaml?
```

This checklist catches the most common modeling errors in financial ML,
especially under team churn where institutional knowledge about data quirks may
be lost.

---

## 8. Agent Workflow Summary

1. Create a feature branch.
2. Formulate your hypothesis. Fill in `configs/reasoning.yaml`.
3. Modify code in `src/` and/or config in `configs/experiment.yaml`.
4. Complete the **Modeling Risk Checklist** above.
5. Run the experiment: `python run_experiment.py --config configs/experiment.yaml`
6. Run tests: `uv run pytest tests/`
7. Commit with the MLflow run ID in the commit message.
8. Open a PR with the MLflow run ID in the description.
9. A human reviews and merges.
