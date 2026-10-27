# Training data

- **Source.** Almost all of `data/processed/` is produced by Shaula (`rachzy/shaula`), the upstream detection and
  feature-extraction pipeline: it finds transit candidates in Kepler light curves and computes their features.
  fang does no light-curve processing or downloading; it consumes Shaula's tables as they are.
- **Shape.** One CSV per star, named `<star-name>_<YYYYMMDD>.csv`; everything before the last underscore is the
  `star_id`. `<star-name>_inverted_<YYYYMMDD>.csv` holds false positives from the inverted light curve and keeps
  the same `star_id`, so folds never split a star from its copy. `<star-name>-koi.csv` is the star's KOI catalogue
  extract and is ignored. A blank file means Shaula found no candidates. Each row's `fp_source` says how its
  label was matched and is withheld as supervision; `UNKNOWN` labels are dropped for training.
- **Size.** One row per candidate signal. Today: 219 KIC stars with candidates in `train/` (824 labelled rows, 116
  CONFIRMED, 288 of the 708 false positives from inverted copies), 10 in `test/`, so every metric is still noisy.
- **Columns.** 41 numeric features reach a model (`feature_columns` in `schema.yaml`): orbital and transit shape
  (`period_days`, `duration_hours`, `vshape_metric`, `ingress_egress_asymmetry`), signal strength (`MES`, `SES_*`,
  `snr_global`, `cdpp_*`), noise and autocorrelation (`local_noise`, `acf_lag_*`), false-positive diagnostics
  (`secondary_depth*`, `odd_even_depth_ratio`), eclipsing-binary checks added 2026-10-27 (`log_rho_ratio`, the
  transit-implied over catalogue stellar density; `even_odd_depth_diff_sigma`; `secondary_max_snr_any_phase`,
  `secondary_phase`, `secondary_depth_ratio` from a secondary search at all phases) and `planet_radius_rearth`. Everything else Shaula emits is listed
  under `excluded_columns` with the reason (labels, pipeline settings, `t0`, unit-converted aliases); an unknown
  column is rejected until the schema is updated.
- **Labels.** `candidate_label` maps `CONFIRMED -> 1`, `FALSE-POSITIVE -> 0`, from the literature cross-match.
  `detection_status` (`accepted`, `rejected`, `provisional`, `harmonic_duplicate`) is Shaula's own verdict and is
  never a feature. The KIC dataset is imbalanced towards false positives (about 6:1); the ratio of FP to TP is a known weak point.
- Every labelled row feeds base learners, the meta-model, threshold selection and reported metrics alike, because
  `predict` scores every row too. An earlier rule that dropped rows below `mes_threshold_used` from the meta stage
  was removed once those rows turned out to be almost all false positives.
- `NaN` is allowed and imputed inside folds; infinities, duplicate rows and empty features are rejected at load.
