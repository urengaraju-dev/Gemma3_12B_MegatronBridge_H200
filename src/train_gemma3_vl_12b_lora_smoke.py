#!/usr/bin/env python
# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
# train_gemma3_vl_12b_lora_smoke.py
#
# END-TO-END SMOKE TEST (NOT a real training run) of a LoRA / PEFT finetune of
# google/gemma-3-12b (a MULTIMODAL Gemma3ForConditionalGeneration checkpoint)
# with NVIDIA Megatron-Bridge on a SINGLE H200.
#
# Goal: prove the whole pipeline runs with NO errors in a few iterations:
#   HF gated checkpoint  ->  AutoBridge HF->Megatron conversion (real 12B weights)
#   ->  LoRA adapter insertion  ->  VLM forward/backward/optimizer  ->  save.
#
# WHY THIS SHAPE (verified against the *installed* Megatron-Bridge 0.6.1):
#   * gemma-3-12b's config declares architecture `Gemma3ForConditionalGeneration`,
#     so AutoBridge yields a `Gemma3VLModelProvider` (vision + text). There is no
#     text-only 12B Gemma3; the supported single-GPU LoRA path is the VL recipe.
#   * VL models use `megatron.bridge.training.vlm_step.forward_step`
#     (confirmed by tests/.../test_recipe_metadata.py: gemma3_vl_* -> "vlm_step").
#   * ConfigContainer.validate() asserts: `if peft is not None: assert
#     checkpoint.pretrained_checkpoint is not None` AND the path must exist on
#     disk. So a LoRA run REQUIRES a real base checkpoint -> we point it at the
#     locally-cached HF snapshot dir, which `_load_hf_pretrained_checkpoint`
#     converts (HF->Megatron) to load the real 12B weights.
#   * The dataset is a synthetic in-memory VLM provider (mirrors the repo's own
#     functional test tests/.../energon/test_hf_task_encoder.py) so the smoke
#     test needs NO dataset download and no images on disk.
#
# Launch (inside the NeMo 26.08 container, 1 GPU):
#   torchrun --nproc-per-node=1 /workspace/scripts/train_gemma3_vl_12b_lora_smoke.py
# =============================================================================
import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _peft_common_vlm
from megatron.bridge.recipes.utils.dataset_utils import default_peft_config
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.vlm_step import forward_step as vlm_forward_step

from megatron.bridge.data.base import DatasetBuildContext, DatasetProvider
from megatron.bridge.data.energon.hf_task_encoder import HFTaskEncoder
from megatron.bridge.data.energon.task_encoder_utils import ChatMLSample

# ----------------------------------------------------------------------------
# Config knobs (smoke test)
# ----------------------------------------------------------------------------
HF_MODEL_ID = os.environ.get("GEMMA_MODEL_ID", "google/gemma-3-12b-pt")   # multimodal base weights
# The base (-pt) processor has NO chat template; the mock builds chat conversations
# that gemma3_vl_collate_fn formats via apply_chat_template. The instruct (-it)
# processor ships a chat template and an identical tokenizer, so use it for
# tokenization/formatting only (AutoProcessor pulls a few MB, no weights).
HF_PROCESSOR_ID = os.environ.get("GEMMA_PROCESSOR_ID", "google/gemma-3-12b-it")
SEQ_LEN = int(os.environ.get("SMOKE_SEQ_LEN", "512"))       # short => fast + low memory
TRAIN_ITERS = int(os.environ.get("SMOKE_TRAIN_ITERS", "3")) # a few steps only
OUT_DIR = os.environ.get("SMOKE_OUT_DIR", "/workspace/outputs/gemma3vl_12b_lora_smoke")


def _resolve_local_snapshot(model_id: str) -> str:
    """Return the local HF snapshot dir (must exist on disk) for pretrained_checkpoint."""
    hub_name = "models--" + model_id.replace("/", "--")
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    pattern = os.path.join(hf_home, "hub", hub_name, "snapshots", "*")
    snaps = sorted(glob.glob(pattern))
    if not snaps:
        raise FileNotFoundError(
            f"No cached snapshot for {model_id} under {pattern}. "
            f"Download it first (hf download {model_id})."
        )
    # pick the snapshot that actually has weights
    for s in snaps:
        if glob.glob(os.path.join(s, "*.safetensors")):
            return s
    return snaps[-1]


# ----------------------------------------------------------------------------
# Synthetic VLM dataset provider (mirrors the repo functional test)
# ----------------------------------------------------------------------------
def _make_chatml_sample(key, conversation, imgs=None):
    import dataclasses

    base_fields = {f.name for f in dataclasses.fields(ChatMLSample)}
    kwargs = {"conversation": conversation}
    if "__key__" in base_fields:
        kwargs["__key__"] = key
    if "__subflavors__" in base_fields:
        kwargs["__subflavors__"] = {}
    if "__restore_key__" in base_fields:
        kwargs["__restore_key__"] = ()
    if "__subflavor__" in base_fields:
        kwargs["__subflavor__"] = None
    if imgs is not None:
        kwargs["imgs"] = imgs
    return ChatMLSample(**kwargs)


@dataclass(kw_only=True)
class MockEnergonHFProvider(DatasetProvider):
    """DatasetProvider that yields synthetic ChatML+image batches (no downloads)."""

    seq_length: int
    hf_processor_path: str
    micro_batch_size: int = 1
    dataloader_type: str = "external"
    skip_getting_attention_mask_from_dataset: bool = True

    def _make_samples(self, num_samples: int = 4):
        rng = np.random.default_rng(42)
        samples = []
        for i in range(num_samples):
            img_tensor = torch.from_numpy(rng.random((3, 64, 64), dtype=np.float32))
            conversation = json.dumps(
                [
                    {"role": "user", "content": "<image> Describe this image."},
                    {"role": "assistant", "content": f"This is a synthetic test image number {i}."},
                ]
            )
            samples.append(_make_chatml_sample(key=f"sample_{i}", conversation=conversation, imgs=[img_tensor]))
        return samples

    def build_datasets(self, context: DatasetBuildContext) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        from transformers import AutoProcessor
        from megatron.bridge.models.hf_pretrained.utils import is_safe_repo

        processor = AutoProcessor.from_pretrained(
            self.hf_processor_path,
            trust_remote_code=is_safe_repo(
                trust_remote_code=self.trust_remote_code,
                hf_path=self.hf_processor_path,
            ),
        )
        task_encoder = HFTaskEncoder(
            processor=processor,
            seq_length=self.seq_length,
            visual_keys=("pixel_values",),
        )
        samples = self._make_samples(num_samples=max(self.micro_batch_size * 2, 4))
        encoded = [task_encoder.encode_sample(s) for s in samples]
        batch = task_encoder.batch(encoded[: self.micro_batch_size])
        batch_dict = task_encoder.encode_batch(batch)

        def _infinite_iter():
            while True:
                yield batch_dict

        # (train, valid, test) iterators; test disabled
        return _infinite_iter(), _infinite_iter(), None


# ----------------------------------------------------------------------------
# Build the smoke-test config
# ----------------------------------------------------------------------------
def build_config():
    cfg = _peft_common_vlm()

    # --- PEFT: LoRA ---
    cfg.peft = default_peft_config("lora")

    # --- Model: 12B multimodal provider from the cached checkpoint (no download) ---
    cfg.model = AutoBridge.from_hf_pretrained(HF_MODEL_ID).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = SEQ_LEN
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.cp_comm_type = "a2a"
    # VLM: train everything the recipe trains (LoRA still freezes base linears)
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False
    cfg.model.transformer_impl = "transformer_engine"
    # no cuda graphs for a smoke test
    cfg.model.cuda_graph_impl = "none"

    # --- Training: a few steps only ---
    cfg.train.train_iters = TRAIN_ITERS
    cfg.train.global_batch_size = 1
    cfg.train.micro_batch_size = 1

    # --- Validation: disabled ---
    cfg.validation.eval_interval = TRAIN_ITERS + 1000
    cfg.validation.eval_iters = 0

    # --- Scheduler: no warmup needed for 3 steps ---
    cfg.scheduler.lr_warmup_iters = 0
    cfg.scheduler.lr_decay_iters = TRAIN_ITERS

    # --- Logging every step ---
    cfg.logger.log_interval = 1

    # --- Dataset: synthetic VLM provider (no download) ---
    cfg.dataset = MockEnergonHFProvider(
        seq_length=SEQ_LEN,
        hf_processor_path=HF_PROCESSOR_ID,
        micro_batch_size=cfg.train.micro_batch_size,
    )

    # --- Checkpoint: LoRA REQUIRES a real base checkpoint -> load real 12B weights ---
    snap = _resolve_local_snapshot(HF_MODEL_ID)
    print(f"[smoke] pretrained_checkpoint (local HF snapshot) = {snap}", flush=True)
    cfg.checkpoint.pretrained_checkpoint = snap
    cfg.checkpoint.save = os.path.join(OUT_DIR, "checkpoints")
    cfg.checkpoint.load = os.path.join(OUT_DIR, "checkpoints")
    cfg.checkpoint.save_interval = TRAIN_ITERS  # save once, at the end

    return cfg


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = build_config()

    # Cheap sanity dump (imports + config assembly), no weight load / no training.
    print("[smoke] config built OK:", flush=True)
    print("   model provider:", type(cfg.model).__name__, flush=True)
    print("   peft:", type(cfg.peft).__name__,
          "| target_modules:", getattr(cfg.peft, "target_modules", None), flush=True)
    print("   vocab_size:", getattr(cfg.model, "vocab_size", None),
          "| num_layers:", getattr(cfg.model, "num_layers", None),
          "| hidden:", getattr(cfg.model, "hidden_size", None), flush=True)
    print("   seq_length:", cfg.model.seq_length,
          "| train_iters:", cfg.train.train_iters,
          "| gbs/mbs:", cfg.train.global_batch_size, cfg.train.micro_batch_size, flush=True)
    print("   pretrained_checkpoint:", cfg.checkpoint.pretrained_checkpoint, flush=True)
    print("   dataset provider:", type(cfg.dataset).__name__, flush=True)

    if os.environ.get("SMOKE_DRY") == "1":
        print("[smoke] SMOKE_DRY=1 -> stopping before training (dry run OK).", flush=True)
        raise SystemExit(0)

    print("[smoke] launching finetune() for Gemma3-VL 12B LoRA "
          f"(seq={SEQ_LEN}, iters={TRAIN_ITERS}, TP=PP=CP=1) ...", flush=True)
    finetune(config=cfg, forward_step_func=vlm_forward_step)
    print("[smoke] DONE — pipeline ran end-to-end with no errors.", flush=True)
