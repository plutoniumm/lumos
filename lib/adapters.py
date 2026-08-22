"""The LoRA adapter as an object with an on/off switch and a strength dial.

Why the switch is worth a class
-------------------------------
peft applies an adapter as ``scaling * B @ A @ x``, and both of the things you
want to do to it -- turn it off, turn it down -- are context managers over
mutable global state on the model. Two bugs in this project came out of that,
and both were the same bug:

* ``src/lora_ladder.py`` measured conditional perplexity *outside* the
  ``scaled()`` block, which restores full adapter strength on exit. So the
  adapter was asked how likely it found its own output. It answered 0.58 where
  the true ratio was 1.80.
* the same ladder scored every rung against a baseline the adapter had also
  seen.

Conditional perplexity is defined against a *reference* model. If the adapter is
live while it is measured, the number is not a perplexity ratio, and nothing
about its value says so -- it just comes out near 1.0 and looks like a clean
result.

So the rule is structural here rather than remembered: ``Adapter.scoring()``
is the only context scoring runs in, and it disables the adapter unconditionally.
An arm cannot rate its own output without deleting that call.
"""

from __future__ import annotations

import json
from contextlib import contextmanager, nullcontext
from pathlib import Path


class Adapter:
    """A loaded LoRA, or nothing at all.

    ``Adapter(None)`` is a working no-op, so a caller that may or may not have
    an adapter does not need a branch at every use site.
    """

    def __init__(self, peft_model=None, path: Path | None = None, meta: dict | None = None):
        self.model = peft_model
        self.path = path
        self.meta = meta or {}

    def __bool__(self) -> bool:
        return self.model is not None

    @classmethod
    def load(cls, lm, path: Path, *, expect_base: str | None = None) -> "Adapter":
        """Wrap *lm* in the adapter at *path* and return the handle.

        peft rewrites the projection ``Linear``s *inside* the existing decoder
        blocks, so ``lm._layers`` keeps pointing at the same module objects and
        the steering hooks attach exactly where they did before. Only
        ``lm.model`` is swapped, for ``generate()``. Doing it the other way
        round -- wrapping first, then locating layers -- fails: ``PeftModel``
        nests the stack one level deeper than ``models._find_layers`` walks.
        """
        try:
            from peft import PeftModel
        except ImportError:
            raise SystemExit("peft is not installed:  pip install peft")

        meta_path = Path(path) / "train_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        trained_on = meta.get("base_model")
        if expect_base and trained_on and trained_on != expect_base:
            raise SystemExit(
                f"FATAL: --lora {path} was trained on {trained_on!r} but this run "
                f"loads {expect_base!r}; an adapter is only valid on its base model"
            )
        peft_model = PeftModel.from_pretrained(lm.model, str(path))
        peft_model.eval()
        lm.model = peft_model
        return cls(peft_model, Path(path), meta)

    def describe(self) -> str:
        if not self:
            return "no adapter"
        m = self.meta
        before, after = m.get("dev_loss_before"), m.get("dev_loss_after")
        loss = (
            f", dev loss {before:.3f} -> {after:.3f}"
            if isinstance(before, float) and isinstance(after, float)
            else ""
        )
        return f"{self.path} (r={m.get('rank', '?')}, {m.get('n_train', '?')} pairs{loss})"

    @contextmanager
    def engaged(self, on: bool = True):
        """Adapter live for *on*, provably off otherwise."""
        if not self or on:
            yield
            return
        with self.model.disable_adapter():
            yield

    def scoring(self):
        """The only context metrics may be computed in. Adapter always off."""
        return self.engaged(False)

    @contextmanager
    def scaled(self, scale: float):
        """Adapter contribution multiplied by *scale* inside the block.

        ``unscale_layer(None)`` restores the configured ``lora_alpha / r``
        rather than dividing back out, so a ladder cannot accumulate float
        drift across rungs and silently report the wrong curve.
        """
        if not self:
            raise RuntimeError("no adapter loaded; nothing to scale")
        from peft.tuners.lora.layer import LoraLayer

        layers = [m for m in self.model.modules() if isinstance(m, LoraLayer)]
        if not layers:
            raise RuntimeError("no LoRA layers found; is this a PeftModel?")
        for m in layers:
            for name in list(m.scaling):
                m.set_scale(name, scale)
        try:
            yield
        finally:
            for m in layers:
                m.unscale_layer(None)

    def at(self, scale: float):
        """``scaled(scale)``, or a no-op context when *scale* is full strength."""
        return self.scaled(scale) if self else nullcontext()
