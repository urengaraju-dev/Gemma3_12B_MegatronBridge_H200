#!/usr/bin/env bash
# =============================================================================
# scripts/run_smoke.sh — launch the Gemma-3-12B LoRA smoke test (single GPU)
#                        inside the running Megatron-Bridge container.
#
# Prerequisite: scripts/setup.sh has been run (container is up).
# Usage:        bash scripts/run_smoke.sh
# =============================================================================
set -euo pipefail

CONTAINER="${CONTAINER:-mbridge}"
SCRIPT="src/train_gemma3_vl_12b_lora_smoke.py"
LOG="logs/train_$(date +%Y%m%d_%H%M%S).out"

echo "==> Launching: torchrun --nproc-per-node=1 ${SCRIPT}"
echo "    logs -> ${CONTAINER}:/workspace/${LOG}"
docker exec "${CONTAINER}" bash -lc \
  "mkdir -p logs && cd /workspace && torchrun --nproc-per-node=1 ${SCRIPT} 2>&1 | tee ${LOG}"
