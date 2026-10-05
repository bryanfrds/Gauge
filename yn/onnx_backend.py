"""ONNX Runtime backend: same scores as PyTorch, less memory and faster on CPU.

PyTorch is a training framework; we only ever run inference. Exporting the model
once to ONNX (Open Neural Network Exchange, a portable model format) lets the much
smaller ONNX Runtime execute it instead. Measured on the default model:

    PyTorch      709 MB peak, 4.0 s load, 0.99 s   (imports alone: 329 MB)
    ONNX         608 MB peak, 0.6 s load, 0.27 s   (imports alone:  46 MB)

Scores agree with the torch backend within 0.002 (largest measured difference:
0.0013). They are not bit-identical: the graphs are optimized and fused differently.
The answer itself did not change in testing, but a confidence within 0.002 of
YN_THRESHOLD could fall either side of `sure`.

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
# Bump when the export's shape changes: input names, opset, tokenizer settings.
# An export from older code is then treated as absent rather than trusted.
FORMAT_VERSION = 1
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


EXPORT_FILES = frozenset({MODEL_FILE, WEIGHTS_FILE, TOKENIZER_FILE, META_FILE})
TEMP_PREFIX = ".export-"
# A temp directory younger than this may belong to an export still running.
STALE_AFTER_SECONDS = 3600


def _is_export_dir(directory: Path) -> bool:
    """True only for a directory YN wrote: its metadata, and nothing but its files.

    `meta.json` alone proves nothing (it is a common name), so the metadata must be
    ours and there must be no file an export doesn't contain.
    """
    meta = read_meta(directory)
    if not isinstance(meta, dict) or "format_version" not in meta or "model" not in meta:
        return False
    return {p.name for p in directory.iterdir()} <= EXPORT_FILES


def _replace_export(tmp: Path, final: Path) -> None:
    """Move `tmp` onto `final`, never deleting anything YN did not write.

    `final` can be any path the user passed to --out. Only an empty directory or an
    earlier export is replaced; anything else is refused. The old export is moved into
    a temp directory made just for it, so no existing path is ever deleted.
    """
    import shutil
    import tempfile

    if not final.exists():
        os.replace(tmp, final)
        return
    if not final.is_dir():
        raise RuntimeError(f"{final} exists and is not a directory")
    if any(final.iterdir()) and not _is_export_dir(final):
        raise RuntimeError(
            f"{final} is not empty and is not a yn export. Refusing to replace it; "
            f"pick an empty directory or remove it yourself."
        )
    # Moved aside rather than deleted first, so a failed swap can put it back. If the
    # process is killed mid-swap, the old export is left in this temp directory and
    # "auto" uses torch until the next export.
    holder = Path(tempfile.mkdtemp(prefix=TEMP_PREFIX + "old-", dir=final.parent))
    old = holder / "export"
    os.replace(final, old)
    try:
        os.replace(tmp, final)
    except OSError:
        os.replace(old, final)  # put it back
        raise
    shutil.rmtree(holder, ignore_errors=True)


def _sweep_stale_temp_dirs(parent: Path) -> None:
    """Remove temp directories a killed export left behind; each is most of a gigabyte.

    Only in YN's own export cache, never next to a user's --out directory, and only
    ones old enough that no export can still be writing to them.
    """
    import shutil
    import time

    if parent.resolve() != export_dir("x").parent.resolve():
        return
    cutoff = time.time() - STALE_AFTER_SECONDS
    for d in parent.glob(TEMP_PREFIX + "*"):
        try:
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass  # removed by someone else meanwhile


def read_meta(directory: Path) -> dict | None:
    """The export's metadata, or None if it is missing or unreadable."""
    try:
        return json.loads((directory / META_FILE).read_text())
    except (OSError, ValueError):
        return None


def is_exported(model_name: str) -> bool:
    """True only for a complete, current export of exactly this model.

    Every file the runtime opens is checked, not just some of them: a partial
    export would otherwise be picked up by the "auto" backend and fail on every
    call instead of falling back to torch. The recorded model name is checked
    too, because _slug() is lossy — "org/model" and "org_model" share a
    directory — and scoring against another model's weights would look like a
    quality problem rather than a bug.
    """
    d = export_dir(model_name)
    if not all((d / f).is_file() for f in (MODEL_FILE, WEIGHTS_FILE, TOKENIZER_FILE)):
        return False
    meta = read_meta(d)
    return (
        meta is not None
        and meta.get("format_version") == FORMAT_VERSION
        and meta.get("model") == model_name
    )


def export(model_name: str, out_dir: Path | None = None) -> Path:
    """Export `model_name` to ONNX. Needs torch, transformers, onnx and onnxscript.

    Writes to a sibling temp directory and moves it into place, so an interrupted
    export never leaves a half-written directory that `is_exported` would trust.
    """
    import shutil
    import tempfile

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    final = Path(out_dir) if out_dir else export_dir(model_name)
    final.parent.mkdir(parents=True, exist_ok=True)
    _sweep_stale_temp_dirs(final.parent)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
    labels = {v.lower(): int(k) for k, v in model.config.id2label.items()}
    if "entailment" not in labels:
        raise RuntimeError(f"{model_name} has no 'entailment' label: {labels}")
    # We export input_ids and attention_mask only. A model with segment embeddings
    # also needs token_type_ids, and baking them to zeros would score plausibly but
    # wrongly, so refuse rather than produce a quietly broken export.
    if getattr(model.config, "type_vocab_size", 0):
        raise RuntimeError(
            f"{model_name} uses token_type_ids (type_vocab_size="
            f"{model.config.type_vocab_size}); the ONNX export supports only models "
            f"without segment embeddings. Use YN_BACKEND=torch for this model."
        )

    tmp = Path(tempfile.mkdtemp(prefix=TEMP_PREFIX, dir=final.parent))
    try:
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
            str(tmp / MODEL_FILE),
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
        tokenizer.backend_tokenizer.save(str(tmp / TOKENIZER_FILE))
        (tmp / META_FILE).write_text(
            json.dumps({
                "format_version": FORMAT_VERSION,
                "model": model_name,
                "entail_idx": labels["entailment"],
                "num_labels": len(labels),
                "opset": OPSET,
                # Recorded rather than assumed: it is 0 for DeBERTa but 1 for the
                # RoBERTa family, and padding with the wrong id skews every score.
                "pad_id": tokenizer.pad_token_id,
                "pad_token": tokenizer.pad_token,
            }, indent=2) + "\n"
        )
        _replace_export(tmp, final)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return final


class OnnxRunner:
    """Loaded ONNX model plus its tokenizer. Mirrors what Decider needs from torch."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.dir = export_dir(model_name)
        missing = [f for f in (MODEL_FILE, WEIGHTS_FILE, TOKENIZER_FILE, META_FILE)
                   if not (self.dir / f).is_file()]
        if missing:
            raise FileNotFoundError(
                f"no ONNX export for {model_name} at {self.dir} "
                f"(missing: {', '.join(missing)}). Run: yn export-onnx"
            )
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._np = np
        meta = read_meta(self.dir) or {}
        if meta.get("format_version") != FORMAT_VERSION:
            raise RuntimeError(
                f"the export at {self.dir} is format {meta.get('format_version')!r}, "
                f"this build needs {FORMAT_VERSION}. Re-run: yn export-onnx"
            )
        if meta.get("model") != model_name:
            raise RuntimeError(
                f"the export at {self.dir} is for {meta.get('model')!r}, not "
                f"{model_name!r}. Re-run: yn export-onnx"
            )
        self.entail_idx = int(meta["entail_idx"])

        self.tokenizer = Tokenizer.from_file(str(self.dir / TOKENIZER_FILE))
        # only_first matches the torch path (yn/model.py). The library default,
        # longest_first, would clip the claim or option as well as the text, and
        # a truncated claim is a different question with a different answer.
        self.tokenizer.enable_truncation(512, strategy="only_first")
        pad_id = meta.get("pad_id")
        if pad_id is None:
            self.tokenizer.enable_padding()
        else:
            self.tokenizer.enable_padding(pad_id=int(pad_id),
                                          pad_token=meta.get("pad_token") or "[PAD]")
        # A second tokenizer for counting only. tokenizer.json carries the truncation
        # settings baked in at export time, so a fresh copy truncates too and must be
        # told not to - otherwise every over-limit claim is reported as exactly 512.
        self._counter = Tokenizer.from_file(str(self.dir / TOKENIZER_FILE))
        self._counter.no_truncation()
        self._counter.no_padding()

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
        """Real token count, untruncated, so an over-limit claim reports its true size."""
        return len(self._counter.encode(text, add_special_tokens=False).ids)

    def logits(self, pairs: list[tuple[str, str]]):
        """Raw logits as a numpy array, shape (len(pairs), num_labels)."""
        np = self._np
        encs = self.tokenizer.encode_batch(pairs)
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        out = self.session.run(None, {"input_ids": ids, "attention_mask": mask})[0]
        return out.astype(np.float32)
