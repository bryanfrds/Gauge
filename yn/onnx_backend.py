"""ONNX Runtime backend: same scores as PyTorch, less memory and faster on CPU.

PyTorch is a training framework; we only ever run inference. Exporting the model
once to ONNX (Open Neural Network Exchange, a portable model format) lets the much
smaller ONNX Runtime execute it instead. Measured on the default model:

    PyTorch      709 MB peak, 4.0 s load, 0.99 s   (imports alone: 329 MB)
    ONNX         608 MB peak, 0.6 s load, 0.27 s   (imports alone:  46 MB)

Scores match to four decimal places, which is the precision callers see.

Export needs `onnx` and `onnxscript` alongside torch; running needs only
`onnxruntime`, `tokenizers` and `numpy`. Both are optional extras, so a plain
install keeps working on PyTorch.

Not quantized: int8 dynamic quantization was measured and rejected. It raised peak
memory to 864 MB (weights are dequantized per run) and wrecked accuracy —
DeBERTa's disentangled attention does not survive it. Do not add it back without
re-measuring both.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

MODEL_FILE = "model.onnx"
WEIGHTS_FILE = "model.onnx.data"  # sidecar the runtime memory-maps
TOKENIZER_FILE = "tokenizer.json"
META_FILE = "meta.json"
# Opset 17 covers every operator the NLI models use and is what current
# onnxruntime releases are tested against.
OPSET = 17


def _slug(model_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name).strip("_")
    # "." and ".." survive the character filter and would point at (or above) the
    # cache root instead of a subdirectory of it. So would a name with nothing safe
    # left in it, e.g. one written entirely in a non-Latin script.
    if not slug.strip("._-"):
        return "model"
    return slug


def export_dir(model_name: str) -> Path:
    """Where an exported model for `model_name` lives.

    YN_ONNX_DIR overrides the cache root, so a container can ship a pre-exported
    model on a read-only volume instead of exporting at start-up.
    """
    root = os.environ.get("YN_ONNX_DIR")
    if root:
        return Path(root).expanduser() / _slug(model_name)
    cache = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(cache).expanduser() / "yn" / "onnx" / _slug(model_name)


def is_exported(model_name: str) -> bool:
    d = export_dir(model_name)
    return (d / MODEL_FILE).is_file() and (d / TOKENIZER_FILE).is_file()


def export(model_name: str, out_dir: Path | None = None) -> Path:
    """Export `model_name` to ONNX. Needs torch, transformers, onnx and onnxscript."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out = Path(out_dir) if out_dir else export_dir(model_name)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
    labels = {v.lower(): int(k) for k, v in model.config.id2label.items()}
    if "entailment" not in labels:
        raise RuntimeError(f"{model_name} has no 'entailment' label: {labels}")

    enc = tokenizer(
        ["sample text"], ["a sample statement"],
        truncation="only_first", max_length=512, padding=True, return_tensors="pt",
    )
    # The default (dynamo) exporter writes weights to a sidecar `model.onnx.data`,
    # which ONNX Runtime memory-maps instead of copying into RSS. Measured: 608 MB
    # peak this way versus 990 MB for a single self-contained file (dynamo=False).
    torch.onnx.export(
        model,
        (enc["input_ids"], enc["attention_mask"]),
        str(out / MODEL_FILE),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=OPSET,
        do_constant_folding=True,
    )
    # The fast tokenizer as one file, so the runtime path needs no transformers.
    tokenizer.backend_tokenizer.save(str(out / TOKENIZER_FILE))
    (out / META_FILE).write_text(
        json.dumps({"model": model_name, "entail_idx": labels["entailment"],
                    "num_labels": len(labels), "opset": OPSET}, indent=2) + "\n"
    )
    return out


class OnnxRunner:
    """Loaded ONNX model plus its tokenizer. Mirrors what Decider needs from torch."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.dir = export_dir(model_name)
        if not is_exported(model_name):
            raise FileNotFoundError(
                f"no ONNX export for {model_name} at {self.dir}. "
                f"Run: yn export-onnx"
            )
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._np = np
        meta = json.loads((self.dir / META_FILE).read_text())
        self.entail_idx = int(meta["entail_idx"])

        self.tokenizer = Tokenizer.from_file(str(self.dir / TOKENIZER_FILE))
        self.tokenizer.enable_truncation(512)
        self.tokenizer.enable_padding()

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if not os.environ.get("YN_VERBOSE"):
            # Graph-optimization notes about nodes it can't constant-fold are normal
            # and per-session; 3 = errors only, matching _quiet_libraries() for torch.
            opts.log_severity_level = 3
        # One session per process; threads are capped so several YN processes on a
        # small box don't each grab every core.
        threads = os.environ.get("YN_ONNX_THREADS")
        if threads:
            opts.intra_op_num_threads = int(threads)
        self.session = ort.InferenceSession(
            str(self.dir / MODEL_FILE), opts, providers=["CPUExecutionProvider"]
        )

    def count_tokens(self, text: str) -> int:
        """Token count without special tokens, matching the tokenizer's own count."""
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def logits(self, pairs: list[tuple[str, str]]):
        """Raw logits as a numpy array, shape (len(pairs), num_labels)."""
        np = self._np
        encs = self.tokenizer.encode_batch([(p, h) for p, h in pairs])
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        out = self.session.run(None, {"input_ids": ids, "attention_mask": mask})[0]
        return out.astype(np.float32)
