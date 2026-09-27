# Design notes

This document records the technical decisions behind the pipeline and the
non-obvious facts that shaped it. All facts were verified against the **installed**
Megatron-Bridge (0.6.1) inside `nvcr.io/nvidia/nemo:26.08`, not against docs alone.

## 1. Model identity: "Gemma 4 12B" → Gemma 3 12B (multimodal)

There is **no 12B in Gemma 4** (Gemma 4 ships as a 26B‑A4B MoE / 31B dense).
**12B is a Gemma 3 size**, so this project targets `google/gemma-3-12b`.

Critically, Gemma‑3‑12B is a **multimodal** checkpoint — its `config.json` declares
`architectures: ["Gemma3ForConditionalGeneration"]` with both a `text_config`
(48 layers, hidden 3840, vocab 262208) and a `vision_config`. Consequently:

- `AutoBridge.from_hf_pretrained("google/gemma-3-12b-pt").to_megatron_provider()`
  returns a **`Gemma3VLModelProvider`** (MRO: `Gemma3VLModelProvider →
  Gemma3ModelProvider → GPTModelProvider`), i.e. a vision+text model.
- There is **no text-only 12B** Gemma‑3 checkpoint, and `from_hf_pretrained` in
  0.6.1 has **no `text_only`** knob. The supported single‑GPU LoRA path is
  therefore the **VLM path**, not a text GPT path.

## 2. Forward step: `vlm_step`, not `gpt_step`

Text GPT models train with `megatron.bridge.training.gpt_step.forward_step`.
VLM recipes (including `gemma3_vl_*`) use
`megatron.bridge.training.vlm_step.forward_step` — confirmed by the repo's own
`tests/unit_tests/scripts/training/test_recipe_metadata.py`
(`gemma3_vl_4b_sft_config -> "vlm_step"`).

## 3. LoRA requires a real base checkpoint

`ConfigContainer.validate()` asserts:

```python
if self.peft is not None:
    assert self.checkpoint.pretrained_checkpoint is not None, \
        "PEFT requires a pretrained checkpoint path"
```

and the path must exist on disk. So a LoRA run **cannot** start from random weights.
We point `cfg.checkpoint.pretrained_checkpoint` at the **local HF snapshot
directory** of the cached checkpoint; Megatron-Bridge's
`_load_hf_pretrained_checkpoint` converts HF → Megatron on load, exercising the
real 12B weight conversion (the heart of Megatron-Bridge).

## 4. Processor: `-it` for the chat template, `-pt` for the weights

The base (`-pt`) processor has **no chat template**, so
`gemma3_vl_collate_fn` → `apply_chat_template` raises
`ValueError: ... does not have a chat template`. The instruct (`-it`) processor
ships a chat template and an **identical tokenizer**, so we use `-it` for
tokenization/formatting only (a few‑MB download, no weights) while keeping the
`-pt` weights for the model.

## 5. Dataset: synthetic in-memory VLM provider

To keep the smoke test hermetic (no dataset download, no images on disk), the
dataset is a synthetic `MockEnergonHFProvider` that generates ChatML
conversations with random images and encodes them through `HFTaskEncoder`. This
mirrors the repo's own functional test
`tests/functional_tests/test_groups/data/energon/test_hf_task_encoder.py`,
which is the canonical CI smoke test for the Gemma3‑VL training path.

For a real run, swap `cfg.dataset` for the shipped recipe dataset — e.g.
`gemma3_vl_12b_peft_1gpu_h100_bf16_config("lora")` uses the `cord_v2` image+text
dataset (`DirectHFSFTDatasetConfig`) — or point it at your own HF/local source.

## 6. Parallelism & precision

Single H200 ⇒ `TP=1, PP=1, CP=1`, `sequence_parallel=False`, bf16 mixed
precision, CUDA graphs disabled. LoRA freezes the 12B base and trains only the
adapters, so optimizer state is negligible and the run fits comfortably in
143 GB.

## 7. Verified result

`docs/results/smoke_run.log` — model built, real 12B weights converted
(12,300,571,248 params), LoRA adapters inserted, 3 iterations (lm loss
4.484 → 3.516 → 2.324, no NaNs), checkpoint saved in `torch_dist` format.
