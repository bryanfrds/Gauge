"""Scoring engine.

Stand-in version: uses an existing open zero-shot NLI model (natural-language
inference: "does this text support this statement?") until YN's own model is trained.
The inputs and outputs match the spec, so swapping the model changes nothing for
callers.
"""

from __future__ import annotations

import os
import threading
from dataclasses import asdict, dataclass

DEFAULT_MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
DEFAULT_THRESHOLD = 0.85
DEFAULT_TEMPLATE = "This text is about {}."
MAX_OPTIONS = 50
MAX_INPUTS = 1000
MAX_PAIRS = 20_000  # inputs x options; roughly 30 s on an Apple Silicon GPU
MAX_INPUT_CHARS = 20_000
MAX_STATEMENT_CHARS = 1_000  # claims and options must fit in the model's 512 tokens
BATCH_SIZE = 16


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
    """YN_THRESHOLD is server config, so a bad value is a RuntimeError, not a caller error."""
    raw = os.environ.get("YN_THRESHOLD")
    if raw is None:
        return DEFAULT_THRESHOLD
    try:
        value = float(raw)
    except ValueError:
        value = -1.0
    if not 0 <= value <= 1:
        raise RuntimeError(f"YN_THRESHOLD must be a number from 0 to 1, got {raw!r}")
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
    ):
        self.model_name = model_name or os.environ.get("YN_MODEL", DEFAULT_MODEL)
        self.threshold = threshold if threshold is not None else _env_threshold()
        self.device = device or os.environ.get("YN_DEVICE", "auto")
        self.template = template
        self._model = None
        self._tokenizer = None
        self._entail_idx = 0
        self._lock = threading.Lock()  # guards loading
        # Guards inference. MCP runs tool calls in parallel worker threads, and
        # concurrent use of one model on the Apple GPU aborts the whole process.
        self._infer_lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            if not os.environ.get("YN_VERBOSE"):
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
        """Raw logits, shape (len(pairs), num_labels), for (text, statement) pairs."""
        import torch

        self.load()  # outside _infer_lock: load() takes its own lock
        out = []
        with self._infer_lock, torch.inference_mode():
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
                out.append(self._model(**enc).logits.float().cpu())
        return torch.cat(out)

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
        probs = logits.softmax(dim=-1)[:, self._entail_idx].tolist()
        return [self._decision({"true": p, "false": 1 - p}) for p in probs]

    def decide_many(self, texts: list[str], options: list[str]) -> list[Decision]:
        """Pick the best-fitting option for each text."""
        options = _check_options(options)
        texts = _check_texts(texts, per_text=len(options))
        if not texts:
            return []
        statements = [_as_statement(o, self.template) for o in options]
        pairs = [(t, s) for t in texts for s in statements]
        entail = self._logits(pairs)[:, self._entail_idx].view(len(texts), len(options))
        # Softmax across options, so scores sum to 1 (single-choice).
        probs = entail.softmax(dim=-1).tolist()
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
