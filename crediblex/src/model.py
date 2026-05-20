"""
CredibleX v2 Model Architecture.

Backbone: ``xlm-roberta-base`` — a multilingual encoder pre-trained on 100 languages
including Hindi, making it suitable for Indian news content.

PEFT: Parameter-Efficient Fine-Tuning via LoRA (Low-Rank Adaptation) injected on the
``query`` and ``value`` attention projections.  Only ~0.5 % of parameters are trainable
during fine-tuning, massively reducing VRAM pressure.

Input to each head: the CLS-token hidden state (dim 768) concatenated with a scalar
``fc_score`` produced by the external Fact Check API, yielding a 769-dimensional vector.
If ``fc_score`` is not available it defaults to 0.5 (neutral prior).

Three classification heads:
    - ``bias_head``   → 5 classes  (0=Far Left … 4=Far Right)
    - ``fact_head``   → 2 classes  (0=Fake, 1=Real)
    - ``intent_head`` → 3 classes  (0=News, 1=Opinion, 2=Satire)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model, TaskType

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_BACKBONE   = "xlm-roberta-base"
_HIDDEN     = 768   # XLM-R base hidden size
_FC_DIM     = _HIDDEN + 1   # CLS + fc_score scalar  → 769

_LORA_CFG = LoraConfig(
    task_type     = TaskType.FEATURE_EXTRACTION,
    r             = 16,
    lora_alpha    = 32,
    lora_dropout  = 0.1,
    target_modules= ["query", "value"],
    bias          = "none",
)


def _make_head(in_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.1) -> nn.Sequential:
    """
    Build a two-layer classification head: Linear → GELU → Dropout → Linear.

    Args:
        in_dim:      Input feature dimension.
        hidden_dim:  Intermediate projection dimension.
        num_classes: Number of output logits.
        dropout:     Dropout probability applied between layers.

    Returns:
        ``nn.Sequential`` module ready for use as a task head.
    """
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes),
    )


class CredibleXv2(nn.Module):
    """
    Multi-task credibility classifier for Indian news content (CredibleX v2).

    The model wraps ``xlm-roberta-base`` with LoRA adapters and attaches three
    lightweight classification heads on top of the enriched CLS representation.

    Attributes:
        encoder:     XLM-R backbone wrapped with LoRA via PEFT.
        dropout:     Dropout applied to the raw CLS token before head input.
        bias_head:   5-class political bias head.
        fact_head:   2-class factuality head (real vs fake).
        intent_head: 3-class intent head (news / opinion / satire).
    """

    def __init__(
        self,
        backbone: str = _BACKBONE,
        lora_config: LoraConfig = _LORA_CFG,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialise CredibleXv2.

        Args:
            backbone:    HuggingFace model id or local path for the encoder.
            lora_config: ``peft.LoraConfig`` instance controlling adapter rank/alpha.
            dropout:     Dropout probability on the CLS embedding before heads.
        """
        super().__init__()

        # ── Backbone + LoRA ───────────────────────────────────────────────────
        try:
            base_encoder = AutoModel.from_pretrained(backbone)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load backbone '{backbone}'. "
                "Ensure you have internet access or a local copy."
            ) from exc

        self.encoder: nn.Module = get_peft_model(base_encoder, lora_config)
        self.encoder.print_trainable_parameters()

        # ── Shared dropout on CLS ─────────────────────────────────────────────
        self.dropout = nn.Dropout(dropout)

        # ── Task heads ────────────────────────────────────────────────────────
        # Input dim = 768 (CLS) + 1 (fc_score) = 769
        self.bias_head   = _make_head(_FC_DIM, 256, 5,   dropout)
        self.fact_head   = _make_head(_FC_DIM, 128, 2,   dropout)
        self.intent_head = _make_head(_FC_DIM, 128, 3,   dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        fc_score: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids:      ``[B, seq_len]`` token IDs.
            attention_mask: ``[B, seq_len]`` binary mask (1 = real token, 0 = pad).
            fc_score:       ``[B]`` float tensor of Fact Check API scores in [0, 1].
                            If ``None``, defaults to 0.5 (neutral) for every sample.

        Returns:
            dict with keys ``"bias"``, ``"factuality"``, ``"intent"`` each mapping
            to a raw logit tensor of shape ``[B, num_classes]``.
        """
        batch_size = input_ids.size(0)

        # ── Encoder ───────────────────────────────────────────────────────────
        try:
            encoder_out = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                logger.error("CUDA OOM in encoder forward. Consider reducing batch_size.")
                raise
            raise

        cls_emb = self.dropout(encoder_out.last_hidden_state[:, 0, :])  # [B, 768]

        # ── Concatenate fc_score ──────────────────────────────────────────────
        if fc_score is None:
            fc_score = torch.full(
                (batch_size, 1),
                0.5,
                dtype=cls_emb.dtype,
                device=cls_emb.device,
            )
        else:
            # Ensure shape is [B, 1]
            fc_score = fc_score.float().to(cls_emb.device).view(batch_size, 1)

        enriched = torch.cat([cls_emb, fc_score], dim=1)  # [B, 769]

        # ── Heads ─────────────────────────────────────────────────────────────
        return {
            "bias":        self.bias_head(enriched),    # [B, 5]
            "factuality":  self.fact_head(enriched),    # [B, 2]
            "intent":      self.intent_head(enriched),  # [B, 3]
        }


def load_checkpoint(
    path: str,
    backbone: str = _BACKBONE,
    device: str = "cpu",
) -> CredibleXv2:
    """
    Load a CredibleXv2 model from a saved checkpoint.

    The checkpoint is expected to be a dict produced by :func:`torch.save` with
    at minimum a ``"model_state"`` key.

    Args:
        path:     Path to the ``.pth`` checkpoint file.
        backbone: Encoder model id (must match the checkpoint's backbone).
        device:   Device string or ``torch.device`` to map tensors onto.

    Returns:
        Initialised ``CredibleXv2`` with weights loaded and set to eval mode.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        KeyError:          If the checkpoint dict is missing ``"model_state"``.
    """
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device)
    if "model_state" not in ckpt:
        raise KeyError(
            f"Checkpoint at '{path}' missing 'model_state' key. "
            f"Keys found: {list(ckpt.keys())}"
        )

    model = CredibleXv2(backbone=backbone)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    logger.info("Loaded CredibleXv2 checkpoint from %s (epoch %s).",
                path, ckpt.get("epoch", "?"))
    return model


if __name__ == "__main__":
    # Quick sanity check — run with:  python src/model.py
    import torch
    print("Building CredibleXv2 …")
    model = CredibleXv2()
    dummy_ids  = torch.zeros(2, 64, dtype=torch.long)
    dummy_mask = torch.ones(2, 64, dtype=torch.long)
    dummy_fc   = torch.tensor([0.8, 0.3])

    out = model(dummy_ids, dummy_mask, dummy_fc)
    print("bias    :", out["bias"].shape)       # [2, 5]
    print("factual :", out["factuality"].shape) # [2, 2]
    print("intent  :", out["intent"].shape)     # [2, 3]
    print("Model sanity check passed.")
