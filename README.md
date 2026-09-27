# Gemma-3-12B LoRA Finetuning on Megatron-Bridge (H200)

![Framework](https://img.shields.io/badge/Megatron--Bridge-0.6.1-76B900)
![Container](https://img.shields.io/badge/NeMo-26.08-76B900)
![CUDA](https://img.shields.io/badge/CUDA-13-76B900)
![GPU](https://img.shields.io/badge/GPU-1%C3%97%20H200%20(143%20GB)-76B900)
![Status](https://img.shields.io/badge/smoke%20test-passing-brightgreen)

A reproducible, single-GPU **LoRA / PEFT finetuning pipeline for Gemma-3-12B**
built on **[NVIDIA Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)**.
It is delivered as an end-to-end **smoke test** — a short, hermetic run that proves
the full pipeline (HF checkpoint → Megatron conversion → LoRA → training loop →
checkpoint) executes without errors on one NVIDIA H200.

---

## Highlights

- **Real weight conversion.** Loads the genuine 12.3B‑parameter checkpoint and
  converts HF → Megatron via `AutoBridge` (not random init).
- **LoRA adapters** on all linear projections (`linear_qkv`, `linear_proj`,
  `linear_fc1`, `linear_fc2`).
- **Hermetic.** Synthetic in-memory dataset — no dataset download, mirroring the
  framework's own CI smoke test.
- **Single H200.** `TP=PP=CP=1`, bf16, fits comfortably in 143 GB.
- **Verified.** loss 4.484 → 3.516 → 2.324 over 3 steps, checkpoint saved.
  ([docs/results/smoke_run.log](docs/results/smoke_run.log))

## Repository layout

```
.
├── README.md                     # this file
├── LICENSE                       # Apache-2.0
├── CHANGELOG.md
├── CONTRIBUTING.md
├── Makefile                      # make setup / smoke / clean
├── configs/
│   └── smoke.env                 # tunable knobs (image, model ids, seq, iters)
├── docs/
│   ├── DESIGN.md                 # technical decisions & verified facts
│   └── results/
│       └── smoke_run.log         # curated evidence of a passing run
├── scripts/
│   ├── setup.sh                  # provision container + weights
│   └── run_smoke.sh              # launch the smoke test
└── src/
    └── train_gemma3_vl_12b_lora_smoke.py   # the pipeline
```

## Requirements

| Component | Version / value |
|---|---|
| GPU | 1× NVIDIA H200 (143 GB) — any ≥ 48 GB CUDA-13 GPU should work for LoRA |
| Container runtime | Docker with the NVIDIA runtime (GPU visible in containers) |
| Container image | `nvcr.io/nvidia/nemo:26.08` (Megatron-Bridge 0.6.1, MCore 0.19.1, CUDA 13) |
| HF access | A HuggingFace token whose account **accepted the Gemma license** at <https://huggingface.co/google/gemma-3-12b-pt> |

## Quickstart

```bash
# 0. Provide an HF token (Gemma license accepted)
echo "hf_xxxxxxxx" > ~/.hf_token && chmod 600 ~/.hf_token   # or: export HF_TOKEN=...

# 1. Provision: pull the container, download weights, start the container
bash scripts/setup.sh

# 2. Run the LoRA smoke test (3 iterations on synthetic data)
bash scripts/run_smoke.sh
```

Or with `make`:

```bash
make setup     # == scripts/setup.sh
make smoke     # == scripts/run_smoke.sh
make clean     # remove the container and local run artifacts
```

## What the pipeline does

`src/train_gemma3_vl_12b_lora_smoke.py`:

1. Builds the config from Megatron-Bridge's VLM PEFT base (`_peft_common_vlm`)
   and attaches a **LoRA** adapter (`default_peft_config("lora")`).
2. Builds the **12B multimodal provider** from the cached checkpoint via
   `AutoBridge.from_hf_pretrained(...).to_megatron_provider(load_weights=False)`,
   set to `TP=1 / PP=1 / CP=1`, bf16, `seq_length=512`.
3. Loads the **real 12B weights** by pointing `checkpoint.pretrained_checkpoint`
   at the local HF snapshot (Megatron-Bridge converts HF → Megatron on load;
   PEFT requires a real base checkpoint).
4. Feeds a **synthetic VLM dataset** (`MockEnergonHFProvider`) so no data is
   downloaded.
5. Runs `finetune(cfg, vlm_step.forward_step)` for 3 iterations and writes a
   `torch_dist` checkpoint.

## Configuration

Tunables live in [`configs/smoke.env`](configs/smoke.env) and are read from the
environment by the trainer (`GEMMA_MODEL_ID`, `GEMMA_PROCESSOR_ID`,
`SMOKE_SEQ_LEN`, `SMOKE_TRAIN_ITERS`, `SMOKE_OUT_DIR`). Override per-run, e.g.:

```bash
docker exec -e SMOKE_TRAIN_ITERS=10 -e SMOKE_SEQ_LEN=1024 mbridge \
  torchrun --nproc-per-node=1 src/train_gemma3_vl_12b_lora_smoke.py
```

## From smoke test to real training

- **More steps / schedule:** raise `SMOKE_TRAIN_ITERS`, restore
  `lr_warmup_iters`, set a real `global_batch_size` with gradient accumulation.
- **Real data:** replace the mock with the shipped recipe dataset — the official
  single-GPU recipe `gemma3_vl_12b_peft_1gpu_h100_bf16_config("lora")` uses the
  `cord_v2` image+text dataset — or point it at your own HF/local VLM source.
- **DoRA instead of LoRA:** change `default_peft_config("lora")` → `("dora")`.
- **Full SFT:** set `peft=None` and use the `gemma3_vl_12b_sft_4gpu_...` recipe
  (multi-GPU; full 12B optimizer state does not fit one H200).
- **FP8 training:** set `mixed_precision="bf16_with_fp8_current_scaling_mixed"`.
- **Merge adapters:** `examples/peft/merge_lora.py` inside the container.

## References

- Megatron-Bridge — <https://github.com/NVIDIA-NeMo/Megatron-Bridge>
- NeMo Framework container — `nvcr.io/nvidia/nemo:26.08`
- Gemma 3 (base) — <https://huggingface.co/google/gemma-3-12b-pt>
- Gemma 3 (instruct) — <https://huggingface.co/google/gemma-3-12b-it>

## License

Apache-2.0 — see [LICENSE](LICENSE). Gemma model weights are governed by the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms); you must accept them on
Hugging Face to download the checkpoints.
