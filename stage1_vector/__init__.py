"""Stage 1 -- full vector baseline (brief section 1).

A sibling package to `lewm/`, not a submodule of it: `lewm`'s dataset layer
is done and should stop changing once something depends on its schema (see
the repo README). This package only *reads* `lewm.dataset.Dataset`; it does
not modify the collector or the schema.

Trains the official LeWM code (github.com/lucas-maes/le-wm, imported
unmodified from a sibling checkout -- see `lewm.paths.add_lewm_official_to_path`)
on our vector observation instead of pixels. The one piece of their code we
cannot reuse as-is is `jepa.JEPA.encode`, which is hardcoded to a ViT pixel
encoder; `model.VectorJEPA` overrides just that method to read the `state`
column through a plain MLP encoder (`module.MLP`, already in their repo)
instead. Everything else -- the autoregressive predictor, the action
embedder, the SIGReg regularizer, the loss -- is theirs, untouched.
"""
