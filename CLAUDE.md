# fang

Star-grouped stacked classifier that screens transit exoplanet candidates. Formerly `transit-exoplanet-ml`.
The package, the distribution and the CLI command are all named `fang` (the import name used to be `src`).
Its output, `potential_probability`, is a screening score learned from literature labels, not a confirmation.

## Layout

- `fang/` - library and CLI (`cli.py`); tests live in `fang/tests/`
- `fang/service.py` - the surface a gRPC service consumes
- `fang/resources/schema.yaml` - the feature schema; `fang/resources/config.yaml` - model, CV and tuning config
- `data/processed/{train,test}/` - per-star candidate CSVs; `data/literature/` - confirmed-planet catalogs (all `data/**/*.csv` are git-ignored)
- `fang/scripts/`, `fang/utils/` - grade predictions / extractions against `data/literature`
- `models/<run_id>/` - training output; `reports/` - evaluation output (both git-ignored)
- `plans/` - local, git-ignored working notes

## Training data

See [docs/training-data.md](docs/training-data.md): source (mostly Shaula, `rachzy/shaula`), file layout,
features and labels.

## The model

The architecture is under active iteration: which base learners run, the meta-learner, the grids and the
selection rule may all change. Treat `fang/resources/config.yaml` as the source of truth and read it before
describing the current model. README and MODEL_CARD lag behind in places (they still name the RBF SVM and logistic
regression as active base learners).

- **Base learners** (`models.py`, registry `_ESTIMATORS`): LightGBM, CatBoost and ExtraTrees are active; RBF SVM
  (Platt-calibrated) and logistic regression are supported but commented out in the config. Each gets its own
  preprocessing (`preprocessing.py`): none for the gradient-boosted models (native `NaN` handling), degenerate-column
  drop plus median imputation with missing indicators for ExtraTrees, plus `RobustScaler` for SVM/LR.
- **Stack** (`stacking.py`, `fit_stack`): an L2 logistic-regression meta-learner over the base learners'
  out-of-fold probabilities, and nothing else. The same routine builds every evaluation fold and the shipped model.
- **Weights.** `star_balanced_weights` gives every star the same total weight in every fit and metric, recomputed
  per subset. No automatic class weighting.
- **Tuning.** Seeded (`seed: 42`), deterministic sampling of `n_candidates` from each grid, scored by inner grouped
  CV on `objective.metric` (average precision).
- **Threshold.** Per model, the highest-precision threshold that keeps recall >= `objective.min_recall` (0.95) on
  cross-fitted training rows.
- **Selection** (`training.py`, `choose_model`): every candidate (`stack` and each base learner) is fitted and kept
  in the bundle; the best average precision among those meeting the recall floor ships as `selected_model`.
  In recent nested evaluations the stack scores below the best gradient-boosted learner; making it outperform them
  without overfitting is an open goal.
- **Evaluation** (`evaluate.py`): nested `StratifiedGroupKFold` grouped by `star_id` (5 outer, 3 inner; never
  below 3/2, else `DataDiversityError`). Never split candidates of one star across folds: they share a light curve.
  Reports star-bootstrap CIs, permutation importance on held-out stars, calibration and per-star tables.

## Libraries

- **scikit-learn** - pipelines, imputation/scaling, `StratifiedGroupKFold`, ExtraTrees, SVC, logistic regression
  (base and meta), `CalibratedClassifierCV`, metrics. Custom transformers follow the sklearn estimator API.
- **LightGBM**, **CatBoost** - gradient-boosted base learners, fed raw features; keep them single-threaded and
  deterministic (`n_jobs`/`thread_count: 1`) so runs reproduce.
- **pandas**, **NumPy**, **SciPy** - loading and validating CSVs (`data.py`), arrays for every fit, metric helpers.
- **joblib** - serialises `ModelBundle` to `model.joblib`. Pickles: only load trusted bundles.
- **PyYAML** - schema, config and run sidecars (`config.resolved.yaml`, `feature_schema.yaml`).
- **Typer** - the CLI in `cli.py`. **matplotlib** - report plots (`plots.py`).
- Dev: **pytest**, **coverage**, **ruff** (rules `E F I UP B SIM RUF`, line length 100). Managed with **uv**;
  `uv.lock` is committed. Python 3.11-3.14.

## Contracts

- **`list_runs()` only returns runs that are safe to serve**: bundle present, run id matching
  `RUN_ID_PATTERN`, sidecars parsing to mappings with a finite numeric threshold, a numeric schema version
  and a non-empty `selected_model`; it also range-checks sidecar values (threshold in [0, 1], int32
  `schema_version`) and requires the run's `schema_version` to equal the installed schema's. Everything else is skipped, with `logger.debug` when sidecars are not
  written yet and `logger.warning` when they are malformed. It must never load `model.joblib`: it sits on a
  request path. This took three rounds of hardening; do not loosen it without reading why
  (`git log -- fang/service.py`).
- **Schema.** `fang/resources/schema.yaml` is the contract with Shaula, the upstream feature-extraction
  pipeline. Bump `schema_version` whenever `feature_columns` changes. `predict_dataset` hard-fails on a
  mismatch between the data and the model.
- **Pre-rename models.** Bundles trained when the import name was `src` cannot be unpickled and must be
  retrained; `ModelBundle.load` reports this as `SchemaVersionError`.
- **Runs are immutable.** Each training run is a directory `models/<run_id>/`. `models/` is git-ignored.

## Git

- Commit messages: `type: brief message` (`feat`, `fix`, `docs`, `refactor`, ...). One commit per logical change.
- No attribution trailers: never add `Co-Authored-By` (or "Co-authored-by") lines or "Generated with" lines.
- Never push.
