# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-09-27

### Added
- Initial single-GPU **LoRA smoke-test pipeline** for Gemma-3-12B on
  Megatron-Bridge (`nvcr.io/nvidia/nemo:26.08`, Megatron-Bridge 0.6.1).
- `src/train_gemma3_vl_12b_lora_smoke.py` — builds the Gemma3-VL 12B provider,
  attaches LoRA, loads real HF weights (HF → Megatron conversion), and runs a
  few training iterations on a synthetic VLM dataset.
- `scripts/setup.sh` and `scripts/run_smoke.sh` — environment provisioning and
  launch.
- `configs/smoke.env` — env-driven tunables.
- `docs/DESIGN.md` — technical decisions and verified facts.
- `docs/results/smoke_run.log` — evidence of a passing run (loss
  4.484 → 3.516 → 2.324 over 3 iterations; checkpoint saved).
- `Makefile`, `CONTRIBUTING.md`, `.gitignore`, `LICENSE` (Apache-2.0).

### Verified
- End-to-end run on 1× NVIDIA H200 (143 GB): 12,300,571,248-parameter model,
  LoRA adapters inserted, no NaNs, `torch_dist` checkpoint written.
