# Training data

- **Source.** Almost all of `data/processed/` is produced by Shaula (`rachzy/shaula`), the upstream detection and
  feature-extraction pipeline: it finds transit candidates in Kepler light curves and computes their features.
  fang does no light-curve processing or downloading; it consumes Shaula's tables as they are.
- **Shape.** One CSV per star, named `<star-name>_<YYYYMMDD>.csv`; everything before the last underscore is the
  `star_id`. One row per candidate signal. Today: 35 Kepler systems in `train/`, 10 in `test/`, a few hundred
  candidates, so every metric is noisy.
- **Columns.** 36 numeric features reach a model (`feature_columns` in `schema.yaml`): orbital and transit shape
  (`period_days`, `duration_hours`, `vshape_metric`, `ingress_egress_asymmetry`), signal strength (`MES`, `SES_*`,
  `snr_global`, `cdpp_*`), noise and autocorrelation (`local_noise`, `acf_lag_*`), false-positive diagnostics
  (`secondary_depth*`, `odd_even_depth_ratio`) and `planet_radius_rearth`. Everything else Shaula emits is listed
  under `excluded_columns` with the reason (labels, pipeline settings, `t0`, unit-converted aliases); an unknown
  column is rejected until the schema is updated.
- **Labels.** `candidate_label` maps `CONFIRMED -> 1`, `FALSE-POSITIVE -> 0`, from the literature cross-match.
  `detection_status` (`accepted`, `rejected`, `provisional`, `harmonic_duplicate`) is Shaula's own verdict and is
  never a feature. The classes are imbalanced towards confirmed planets; the ratio of FP to TP is a known weak point.
- **Reliability.** A row is "reliable" when `MES >= mes_threshold_used`. Unreliable rows still train base learners
  but are kept out of meta-model fitting, threshold selection and reported metrics
  (`objective.exclude_unreliable_from_threshold`).
- `NaN` is allowed and imputed inside folds; infinities, duplicate rows and empty features are rejected at load.
