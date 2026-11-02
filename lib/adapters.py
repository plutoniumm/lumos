"""The LoRA adapter as an object with an on/off switch and a strength dial."""

from __future__ import annotations

import json
from contextlib import contextmanager, nullcontext
from pathlib import Path


class Adapter:
    """A loaded LoRA, or nothing at all.

    ``Adapter(None)`` is a working no-op, so callers don't need a branch at
    every use site.
    """

    def __init__(
        self, peft_model=None, path: Path | None = None, meta: dict | None = None
    ):
        self.model = peft_model
        self.path = path
        self.meta = meta or {}

    def __bool__(self) -> bool:
        return self.model is not None

    @classmethod
    def load(cls, lm, path: Path, *, expect_base: str | None = None) -> "Adapter":
        """Wrap *lm* in the adapter at *path* and return the handle."""
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
        """Adapter contribution multiplied by *scale* inside the block."""
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
