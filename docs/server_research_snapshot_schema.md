# Stage-1 Server Research Snapshot

## Purpose

`scripts/collect_server_research_snapshot.py` collects a compact, read-only
evidence manifest from one Kerr project checkout. It is designed for a user to
run manually on the Linux research server and redirect to a file outside the
project tree.

```bash
python scripts/collect_server_research_snapshot.py \
  --project-root /path/to/kerr_project \
  > /safe/local/location/server_research_snapshot.json
```

The collector never creates the redirected file. The shell performs that
redirection outside the project tree.

## Stage 1 and Stage 2

Stage 1 discovers assets and preserves compact evidence: repository identity,
dataset metadata, run configuration, summaries, metrics, diagnostics, and
small training histories. It does not copy trajectories, predictions, or model
weights.

Stage 2 is a separate, explicitly approved scientific-validation task. It may
perform targeted strong hashing, short/long prefix validation, trajectory
identity checks, and coarse/fine grid mapping only for assets selected for
reuse.

## Allowed roots

The collector accepts `--project-root` and recursively scans only:

- `data/tasks/`
- `outputs/`

It also reads the one allowlisted project-root file
`SERVER_DATA_EXPERIMENT_REGISTRY.md` when present. No other project-root
directory is recursively scanned. The collector does not inspect `.git` as a
filesystem tree.

Repository metadata is obtained only through these read-only commands:

```text
git rev-parse HEAD
git branch --show-current
git status --short
git log -1 --format=%s
git describe --always --dirty
```

## Safety guarantees

The collector uses directory listing, `lstat`, read-only file access, JSON/CSV
parsing, ZIP central-directory inspection, limited `.npy` header parsing,
small parameter-array reads, small-file hashing, safe Git metadata commands,
and stdout output.

It does not create directories or files in the target project, deserialize
checkpoints, import PyTorch, initialize CUDA, use model/training/evaluation
code, run dataset generation, run inference, or execute Git write/remote
commands.

Every path is kept project-relative. Symlinks are recorded but never followed;
symlinked directories are not traversed. A symlink resolving outside the
project root is reported with `outside_allowlist`.

## Output schema

The collector writes one JSON document with `schema_version` set to `"1.0"`.

```json
{
  "schema_version": "1.0",
  "collection": {},
  "repository": {},
  "scan_roots": [],
  "registry": {},
  "datasets": [],
  "runs": [],
  "checkpoints": [],
  "evidence_files": [],
  "legacy_assets": [],
  "errors": []
}
```

`datasets` records each direct non-symlink child of `data/tasks/`. It includes
`meta.json`, minimal `dataset.npz` archive metadata, and `failed_samples.json`
when those files exist. `runs` is a technical inventory inferred from output
artifacts; its status is never a scientific validity classification.

`legacy_assets` is a path inventory for normal files, directories, and
symlinks encountered under the two approved roots. `known_status` is only
derived from explicit registry or metadata evidence; otherwise later review
must treat an asset as `unknown`.

## Thresholds and evidence priority

Default limits are:

```text
embed_file_max_bytes = 2097152
embed_total_soft_max_bytes = 67108864
parameter_array_max_bytes = 1048576
npy_header_max_bytes = 1048576
```

Small embedded files receive a SHA-256 content hash. Large datasets and
checkpoints are not hashed by default.

When the total content budget is reached, the collector continues its
inventory and records `content_omitted_due_to_total_budget`. Priority order is:

1. `SERVER_DATA_EXPERIMENT_REGISTRY.md` and dataset `meta.json`.
2. `run_config.json`, `summary.json`, training summaries, and histories.
3. Metrics, results, comparisons, and diagnostics.
4. Relevant report text and lower-priority legacy evidence.

## NPZ metadata strategy

Current project datasets are written with `np.savez_compressed`. Accessing an
array through `numpy.load(path)["y_train"]` may materialize the full compressed
trajectory array. The collector therefore uses `zipfile` to inspect archive
members and reads only the `.npy` magic/version/header bytes needed to obtain
array name, shape, dtype, `fortran_order`, compressed bytes, and uncompressed
bytes.

It never fully reads `y_train`, `y_val`, `y_test`, predictions, or targets.
For small `x_train`, `x_val`, and `x_test` members only, it may read the full
array when the declared uncompressed member size is at most
`parameter_array_max_bytes`. Q-only arrays embed complete values; other small
parameter arrays retain shape, dtype, per-column bounds, first/last rows, and
a content hash.

## Evidence inclusion

Relevant JSON includes names such as `run_config.json`, `summary.json`,
`train_summary.json`, `train_history.json`, `metrics.json`, `result.json`,
`analysis_summary.json`, diagnostic JSON, comparison JSON, and validation JSON.
Relevant CSV includes per-Q, lambda-profile, trajectory, summary, comparison,
validation, and metrics outputs. Relevant Markdown/text must be under
`outputs/` and have a name containing `report`, `summary`, `validation`, or
`analysis`.

Binary arrays, checkpoint binaries, figures, PDFs, and logs are inventory-only.

## Known limitations

Stage 1 does not prove that two datasets contain identical trajectories, that a
long dataset is an exact short prefix, that coarse and fine grids are paired,
or that any result is scientifically valid. It does not interpret TimesNet or
solver diagnostics. Those are later review and Stage-2 validation tasks.
