"""Gauge: a small decision model. Answers true/false or picks from a list, with confidence."""

from gauge.model import Decider, Decision, get_decider

__version__ = "0.0.1"
__author__ = "bryanfrds"
CREDIT = f"Gauge {__version__}, by bryanfrds (https://github.com/bryanfrds/Gauge)"


def check(text: str, claim: str) -> Decision:
    """Is `claim` true of `text`? Returns answer "true"/"false" with confidence."""
    return get_decider().check(text, claim)


def decide(text: str, options: list[str]) -> Decision:
    """Pick the option that best fits `text`."""
    return get_decider().decide(text, options)


__all__ = ["Decider", "Decision", "check", "decide", "get_decider"]
