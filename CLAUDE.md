# fang

Star-grouped stacked classifier that screens transit exoplanet candidates. Formerly `transit-exoplanet-ml`.
The package, the distribution and the CLI command are all named `fang` (the import name used to be `src`).

## Commands

- `uv sync` - install dependencies
- `uv run pytest` - test suite, currently 221 tests. `fang/tests/test_service.py` is noticeably slower
  than the rest: its `trained_run` fixture trains a real model.
- `uv run ruff check fang` - passes clean today; new code must keep it clean
- `uv run fang --help` - CLI: `validate`, `evaluate`, `train`, `predict`
- `make help` lists the Makefile targets (`test`, `lint`, `train`, `predict`, `full-pipeline`, ...)

## Layout

- `fang/` - library and CLI (`cli.py`); tests live in `fang/tests/`
- `fang/service.py` - the surface a gRPC service consumes
- `fang/resources/schema.yaml` - the feature schema
- `models/<run_id>/` - training output

## Contracts

- **Service surface.** The Antares `apps/fang` gRPC service imports `predict_features()`, `list_runs()` and
  `latest_run()` from `fang/service.py`. Changing their signatures is a breaking change for that service.
  `predict_features` returns the caller's `star_id` as supplied; only the internal staging filename is sanitised.
- **`list_runs()` only returns runs that are safe to serve**: bundle present, run id matching
  `RUN_ID_PATTERN`, sidecars parsing to mappings with a finite numeric threshold, a numeric schema version
  and a non-empty `selected_model`. Everything else is skipped, with `logger.debug` when sidecars are not
  written yet and `logger.warning` when they are malformed. It must never load `model.joblib`: it sits on a
  request path. This took three rounds of hardening; do not loosen it without reading why
  (`git log -- fang/service.py`).
- **Schema.** `fang/resources/schema.yaml` is the contract with Shaula, the upstream feature-extraction
  pipeline. Bump `schema_version` whenever `feature_columns` changes. `predict_dataset` hard-fails on a
  mismatch between the data and the model.
- **Runs are immutable.** Each training run is a directory `models/<run_id>/`. `models/` is git-ignored.

## Git

- Commit messages: `type: brief message` (`feat`, `fix`, `docs`, `refactor`, ...). One commit per logical change.
- No attribution trailers (no `Co-Authored-By`, no "Generated with" lines).
- Never push.
