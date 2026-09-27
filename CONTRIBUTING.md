# Contributing

Thanks for your interest in improving this pipeline.

## Development workflow

1. Branch from `main`: `git checkout -b <type>/<short-description>`
   (`feat/…`, `fix/…`, `docs/…`, `chore/…`).
2. Make focused changes; keep the smoke test green.
3. Verify locally:
   ```bash
   make setup      # once
   make smoke      # must reach "[smoke] DONE — pipeline ran end-to-end"
   ```
4. Open a merge request against `main` with a clear description and, where
   relevant, an updated `docs/results/smoke_run.log` excerpt.

## Conventions

- **Commits:** imperative mood, e.g. `fix: use -it processor for chat template`.
- **Ground truth over docs:** verify Megatron-Bridge APIs against the *installed*
  package inside the container (`import megatron.bridge; inspect.getsource(...)`),
  not just upstream docs — the pinned container version is authoritative here.
- **No secrets in git:** never commit `~/.hf_token`, tokens, or `*.pem`. The
  `.gitignore` already excludes them and all run artifacts (`logs/`, `outputs/`,
  checkpoints, model caches).
- **Keep it reproducible:** pin the container tag; document any new tunable in
  `configs/smoke.env` and read it via the environment in code.

## Reporting issues

Use the project issue tracker. Include the container tag, GPU, the failing
command, and the relevant tail of the training log.
