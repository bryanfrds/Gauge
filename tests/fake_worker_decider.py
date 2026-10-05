"""A Decider stand-in for the worker tests. It runs in a child process, so it
answers with something the test can check came from there: the child's pid."""

from __future__ import annotations

import os

from yn.model import Decision, InputError


class FakeDecider:
    def check(self, text, claim):
        if text == "bad":
            raise InputError("input must be a non-empty string")
        if text == "crash":
            os._exit(3)
        if text == "print":
            print("library noise", flush=True)
        return Decision("true", 0.9, {"true": 0.9, "false": 0.1}, True, f"pid {os.getpid()}")

    def decide(self, text, options):
        return Decision(options[0], 0.8, {o: 0.8 for o in options}, False, "fake")

    def check_many(self, texts, claim):
        return [self.check(t, claim) for t in texts]

    def decide_many(self, texts, options):
        return [self.decide(t, options) for t in texts]

    def check_statements(self, statements):
        if any(len(s) > 10 for s in statements):
            raise InputError("a claim or option is too long")
