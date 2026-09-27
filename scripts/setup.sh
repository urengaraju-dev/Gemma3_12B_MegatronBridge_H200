#!/usr/bin/env bash
# =============================================================================
# scripts/setup.sh — provision the Megatron-Bridge environment for the
#                    Gemma-3-12B LoRA smoke test on a single NVIDIA GPU.
#
# Idempotent. Steps:
#   1. Resolve an HF token (Gemma license must be accepted).
#   2. Pull the NeMo Framework container (ships Megatron-Bridge preinstalled).
#   3. Pre-download the gated gemma-3-12b base weights into a shared HF cache.
#   4. Start a long-lived container with GPU + caches + repo mounted at /workspace.
#   5. Verify megatron.bridge imports and the GPU is visible inside.
#
# Prerequisites:
#   * Docker with the NVIDIA container runtime (GPU visible in containers).
#   * An HF token for an account that has accepted the Gemma license at
#     https://huggingface.co/google/gemma-3-12b-pt  (put it in ~/.hf_token or
#     export HF_TOKEN).
#
# Usage:  bash scripts/setup.sh
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- tunables (override via env) --------------------------------------------
IMAGE="${IMAGE:-nvcr.io/nvidia/nemo:26.08}"           # bundles Megatron-Bridge 0.6.1
CONTAINER="${CONTAINER:-mbridge}"
HF_CACHE="${HF_CACHE:-/ephemeral/cache/huggingface}"  # persistent host HF cache
MODEL_BASE="${MODEL_BASE:-google/gemma-3-12b-pt}"     # multimodal base weights

echo "==> repo    : ${REPO_DIR}"
echo "==> image   : ${IMAGE}"
echo "==> HF cache: ${HF_CACHE}"
mkdir -p "${HF_CACHE}"

# --- 1) HF token ------------------------------------------------------------
if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.hf_token" ]]; then
  HF_TOKEN="$(cat "${HOME}/.hf_token")"
fi
: "${HF_TOKEN:?ERROR: set HF_TOKEN (or ~/.hf_token) for a Gemma-licensed HF account}"
export HF_TOKEN HF_HOME="${HF_CACHE}"

# --- 2) container image -----------------------------------------------------
echo "==> Pulling ${IMAGE} (large, one-time)..."
docker pull "${IMAGE}"

# --- 3) pre-download base weights into the shared cache ---------------------
echo "==> Pre-downloading ${MODEL_BASE} into ${HF_CACHE} ..."
if command -v hf >/dev/null 2>&1; then
  HF_HOME="${HF_CACHE}" hf download "${MODEL_BASE}" --repo-type model
else
  HF_HOME="${HF_CACHE}" python -c \
    "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_BASE}')"
fi

# --- 4) (re)start the long-lived container ----------------------------------
echo "==> (Re)starting container '${CONTAINER}' ..."
docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
docker run -d --name "${CONTAINER}" \
  --gpus all --ipc=host --shm-size=16g \
  -e HF_HOME=/hf_cache \
  -e HF_TOKEN \
  -v "${REPO_DIR}":/workspace \
  -v "${HF_CACHE}":/hf_cache \
  -w /workspace \
  "${IMAGE}" sleep infinity

# --- 5) verify --------------------------------------------------------------
echo "==> Verifying megatron.bridge + GPU inside the container ..."
docker exec "${CONTAINER}" python -c \
  "import torch, importlib.metadata as m; \
   print('megatron-bridge', m.version('megatron-bridge')); \
   print('cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"

echo
echo "==> Environment ready. Run the smoke test with:  bash scripts/run_smoke.sh"
