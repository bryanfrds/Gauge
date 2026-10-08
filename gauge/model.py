# Copyright 2026 bryanfrds (https://github.com/bryanfrds)
# SPDX-License-Identifier: Apache-2.0
"""Scoring engine.

Stand-in version: uses an existing open zero-shot NLI model (natural-language
inference: "does this text support this statement?") until Gauge's own model is trained.
The inputs and outputs match the spec, so swapping the model changes nothing for
callers.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
DEFAULT_THRESHOLD = 0.85
DEFAULT_TEMPLATE = "This text is about {}."
MAX_OPTIONS = 50
MAX_INPUTS = 1000
MAX_PAIRS = 20_000  # inputs x options; roughly 30 s on an Apple Silicon GPU
MAX_INPUT_CHARS = 20_000
MAX_STATEMENT_CHARS = 1_000  # cheap first check, before tokenizing
# Claims and options can't be truncated, and must leave room for the input text in the
# model's 512 tokens. Non-Latin scripts can use a token per character, so this is
# checked in tokens as well as characters.
MAX_STATEMENT_TOKENS = 400
BATCH_SIZE = 16
# "auto" prefers an ONNX export when one exists, else PyTorch. See gauge/onnx_backend.py.
DEFAULT_BACKEND = "auto"
# Written by train/train_choices.py beside a trained model.
CALIBRATION_FILE = "gauge_calibration.json"


def _softmax(a, axis: int = -1):
    """Softmax over `axis` for a numpy array. Shared by both backends."""
    import numpy as np

    shifted = a - a.max(axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=axis, keepdims=True)


class InputError(ValueError):
    """The caller's request is invalid (as opposed to a setup or library problem)."""


@dataclass
class Decision:
    answer: str
    confidence: float
    scores: dict[str, float]
    sure: bool
    model: str

    def to_dict(self) -> dict:
        return asdict(self)


def _as_statement(option: str, template: str) -> str:
    """Options ending in . ! or ? are full statements; bare labels get the template."""
    option = option.strip()
    return option if option[-1] in ".!?" else template.format(option)


def _quiet_libraries() -> None:
    """Hide download bars and library warnings; stderr stays readable for real errors."""
    import warnings

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    warnings.filterwarnings("ignore", category=FutureWarning)
    from huggingface_hub.utils import logging as hub_logging
    from transformers.utils import logging as tf_logging

    hub_logging.set_verbosity_error()
    tf_logging.set_verbosity_error()
    tf_logging.disable_progress_bar()


def _env_threshold() -> float:
    """GAUGE_THRESHOLD is server config, so a bad value is a RuntimeError, not a caller error."""
    raw = os.environ.get("GAUGE_THRESHOLD")
    if raw is None:
        return DEFAULT_THRESHOLD
    try:
        value = float(raw)
    except ValueError:
        value = -1.0
    if not 0 <= value <= 1:
        raise RuntimeError(f"GAUGE_THRESHOLD must be a number from 0 to 1, got {raw!r}")
    return value


def _read_temperature(model_name: str) -> float:
    """The temperature fitted when this model was trained, or 1 (scores as they are).
    Only a local model directory can have one; a hub name never does."""
    path = Path(model_name) / CALIBRATION_FILE
    if not path.is_file():
        return 1.0
    try:
        value = float(json.loads(path.read_text())["temperature"])
    except (OSError, ValueError, KeyError, TypeError):
        value = -1.0
    if not 0 < value < float("inf"):
        raise RuntimeError(f"{path} needs a positive \"temperature\"; retrain, or delete "
                           f"the file to use uncalibrated scores")
    return value


def _check_text(text: str, what: str, max_chars: int = MAX_INPUT_CHARS) -> str:
    if not isinstance(text, str) or not text.strip():
        raise InputError(f"{what} must be a non-empty string")
    if len(text) > max_chars:
        raise InputError(f"{what} is {len(text)} characters; the limit is {max_chars}")
    return text


def _check_texts(texts: list[str], per_text: int = 1) -> list[str]:
    if len(texts) > MAX_INPUTS:
        raise InputError(f"at most {MAX_INPUTS} inputs per call, got {len(texts)}")
    if len(texts) * per_text > MAX_PAIRS:
        raise InputError(f"inputs x options is {len(texts) * per_text}; the limit is "
                         f"{MAX_PAIRS}. Split the inputs across several calls.")
    return [_check_text(t, "input") for t in texts]


def _check_options(options: list[str]) -> list[str]:
    if not 2 <= len(options) <= MAX_OPTIONS:
        raise InputError(f"options must have 2-{MAX_OPTIONS} items, got {len(options)}")
    for o in options:
        _check_text(o, "each option", MAX_STATEMENT_CHARS)
    if len({o.strip() for o in options}) != len(options):
        raise InputError("options must be unique")
    return options


class Decider:
    def __init__(
        self,
        model_name: str | None = None,
        threshold: float | None = None,
        device: str | None = None,
        template: str = DEFAULT_TEMPLATE,
        backend: str | None = None,
    ):
        self.model_name = model_name or os.environ.get("GAUGE_MODEL", DEFAULT_MODEL)
        self.threshold = threshold if threshold is not None else _env_threshold()
        self.device = device or os.environ.get("GAUGE_DEVICE", "auto")
        self.backend = backend or os.environ.get("GAUGE_BACKEND", DEFAULT_BACKEND)
        if self.backend not in ("auto", "torch", "onnx"):
            raise RuntimeError(
                f"GAUGE_BACKEND must be auto, torch or onnx, got {self.backend!r}"
            )
        self.template = template
        # Softens decide()'s scores so a stated 0.9 is right about 90% of the time.
        # It was fitted on decide's softmax across options, so check() doesn't use it.
        self.temperature = _read_temperature(self.model_name)
        self._model = None
        self._runner = None  # ONNX backend, when in use
        self._tokenizer = None
        self._entail_idx = 0
        self._lock = threading.Lock()  # guards loading
        # Guards inference. MCP runs tool calls in parallel worker threads. On torch,
        # concurrent use of one model on the Apple GPU aborts the whole process; on
        # ONNX the session itself is thread-safe but the tokenizer is shared mutable
        # state, so both backends need this held.
        self._infer_lock = threading.Lock()

    def _use_onnx(self) -> bool:
        """Explicit "onnx" always; "auto" only when an export is already on disk."""
        if self.backend == "torch":
            return False
        from gauge.onnx_backend import is_exported

        if self.backend == "onnx":
            return True
        # ONNX runs on the CPU here, so "auto" leaves an explicitly chosen GPU alone.
        if self.device not in ("auto", "cpu"):
            return False
        return is_exported(self.model_name)

    def _load_onnx_runner(self):
        """Build the ONNX runner, or None when "auto" should fall back to torch.

        An export can be complete and current and still not load: onnxruntime may not
        be installed (`pip install gauge-model[export]` alone does exactly that), or the graph
        may be corrupt. "auto" promises a silent fallback, so only "onnx" raises.
        A bad GAUGE_ONNX_THREADS is the user's config, though, so it always raises.
        """
        from gauge.onnx_backend import _env_threads

        _env_threads()
        try:
            from gauge.onnx_backend import OnnxRunner

            return OnnxRunner(self.model_name)
        except Exception as e:
            if self.backend == "onnx":
                raise
            if os.environ.get("GAUGE_VERBOSE"):
                import sys

                print(f"gauge: ONNX backend unavailable ({type(e).__name__}: {e}); "
                      f"using torch", file=sys.stderr)
            return None

    def load(self) -> None:
        with self._lock:
            if self._model is not None or self._runner is not None:
                return
            if self._use_onnx():
                runner = self._load_onnx_runner()
                if runner is not None:
                    self._entail_idx = runner.entail_idx
                    self.device = "cpu"  # ONNX Runtime here is CPU-only
                    self._runner = runner
                    return
                # backend="auto" and the runner wouldn't build: fall through to torch.
            if not os.environ.get("GAUGE_VERBOSE"):
                _quiet_libraries()
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            if self.device == "auto":
                # Apple GPU is ~3x faster than CPU here, with matching scores.
                self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            labels = {v.lower(): int(k) for k, v in model.config.id2label.items()}
            if "entailment" not in labels:
                raise RuntimeError(f"{self.model_name} has no 'entailment' label: {labels}")
            self._entail_idx = labels["entailment"]
            self._model = model.to(self.device).eval()

    def _logits(self, pairs: list[tuple[str, str]]):
        """Raw logits as numpy, shape (len(pairs), num_labels), for (text, statement)."""
        import numpy as np

        self.load()  # outside _infer_lock: load() takes its own lock
        if self._runner is not None:
            with self._infer_lock:
                self._check_statement_tokens({h for _, h in pairs})
                out = [
                    self._runner.logits(pairs[i : i + BATCH_SIZE])
                    for i in range(0, len(pairs), BATCH_SIZE)
                ]
            return np.concatenate(out)

        import torch

        out = []
        with self._infer_lock, torch.inference_mode():
            self._check_statement_tokens({h for _, h in pairs})
            for i in range(0, len(pairs), BATCH_SIZE):
                chunk = pairs[i : i + BATCH_SIZE]
                enc = self._tokenizer(
                    [p for p, _ in chunk],
                    [h for _, h in chunk],
                    truncation="only_first",
                    max_length=512,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                out.append(self._model(**enc).logits.float().cpu().numpy())
        return np.concatenate(out)

    def check_statements(self, statements: list[str]) -> None:
        """Raise InputError if any statement is over the token limit. Loads the model."""
        self.load()
        with self._infer_lock:
            self._check_statement_tokens(set(statements))

    def _check_statement_tokens(self, statements: set[str]) -> None:
        """Call with _infer_lock held (the tokenizer is shared)."""
        for s in statements:
            if self._runner is not None:
                n = self._runner.count_tokens(s)
            else:
                n = len(self._tokenizer(s, add_special_tokens=False)["input_ids"])
            if n > MAX_STATEMENT_TOKENS:
                raise InputError(f"a claim or option is {n} tokens; the limit is "
                                 f"{MAX_STATEMENT_TOKENS}. Shorten it.")

    def _decision(self, scores: dict[str, float]) -> Decision:
        answer = max(scores, key=scores.get)
        conf = round(scores[answer], 4)
        # `sure` uses the same rounded value callers see, so the two never disagree.
        return Decision(answer, conf, {k: round(v, 4) for k, v in scores.items()},
                        conf >= self.threshold, self.model_name)

    def check_many(self, texts: list[str], claim: str) -> list[Decision]:
        """True/false: is `claim` true of each text?"""
        claim = _check_text(claim, "claim", MAX_STATEMENT_CHARS)
        texts = _check_texts(texts)
        if not texts:
            return []
        logits = self._logits([(t, claim) for t in texts])
        # Softmax over the model's own labels; P(true) = P(entailment).
        probs = _softmax(logits)[:, self._entail_idx].tolist()
        return [self._decision({"true": p, "false": 1 - p}) for p in probs]

    def decide_many(self, texts: list[str], options: list[str]) -> list[Decision]:
        """Pick the best-fitting option for each text."""
        options = _check_options(options)
        texts = _check_texts(texts, per_text=len(options))
        if not texts:
            return []
        statements = [_as_statement(o, self.template) for o in options]
        pairs = [(t, s) for t in texts for s in statements]
        entail = self._logits(pairs)[:, self._entail_idx].reshape(len(texts), len(options))
        # Softmax across options, so scores sum to 1 (single-choice).
        probs = _softmax(entail / self.temperature).tolist()
        return [self._decision(dict(zip(options, row))) for row in probs]

    def check(self, text: str, claim: str) -> Decision:
        return self.check_many([text], claim)[0]

    def decide(self, text: str, options: list[str]) -> Decision:
        return self.decide_many([text], options)[0]


_default: Decider | None = None
_default_lock = threading.Lock()


def get_decider() -> Decider:
    """Shared instance, so the model is loaded once per process."""
    global _default
    with _default_lock:
        if _default is None:
            _default = Decider()
        return _default
