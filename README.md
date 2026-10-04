# lewm-car-nav

Latent world model (LeWM) experiments on the 3D car-navigation environment.
**This repository contains no simulator** — it reads
[`Car-Navigation-Env`](../Car-Navigation-Env) from a sibling checkout and never
modifies it.

Current state: **stage 0 complete** — dataset instrumentation, a validation
tool, and a 12-episode pilot that passes 37/37 checks. No model has been
trained yet, deliberately. See [docs/LEWM_DATASET.md](docs/LEWM_DATASET.md) for
the schema, the visual setup and the compatibility report, and
[docs/lewm_pilot/PILOT_REPORT.md](docs/lewm_pilot/PILOT_REPORT.md) for the
pilot's measured numbers.

## Why this is a separate repository

Same reason `dreamer-car-nav` keeps the DreamerV3 integration out of the
environment it trains on: the environment is a shared, stable artifact with its
own test suite, and algorithm work moves fast and is one of several possible
consumers. Mixing them means every world-model experiment becomes a diff
against the simulator, and "did the env change?" stops being answerable by
looking at one repository's log.

The dependency points one way and is enforced by layout: `lewm` imports
`carnav`, `env.car`, `baselines.scripted` and `render.panda_renderer`; nothing
in the env imports `lewm`. The env's whole contribution to this work is one
additive, default-off-by-value keyword (`show_goal_beacon`) on its renderer.

## Finding the simulator

`lewm/paths.py` resolves it at import time, first hit wins:

1. `CARNAV_ROOT`, if set
2. an already-importable `carnav` (editable install or `PYTHONPATH`)
3. `../Car-Navigation-Env`, the sibling checkout

So a fresh clone next to the env repo needs nothing but numpy (plus `panda3d`
for RGB capture). No install step, no path juggling at the call site.

```
world_models/
    Car-Navigation-Env/     the simulator (untouched by this repo)
    lewm-car-nav/           this repo
```

## Quick start

```bash
python -m lewm pilot ~/lewm_runs/pilot_001    # collect, validate, figures, bench, report
python -m lewm collect ~/lewm_runs/pixels --rgb --n-train 200
python -m lewm validate ~/lewm_runs/pixels
python -m lewm bench --rgb
python -m lewm info ~/lewm_runs/pixels
python tests/test_lewm.py                     # 27 checks, incl. env compatibility
```

```python
from lewm import CollectConfig, Dataset, collect

collect(CollectConfig(capture_rgb=True, n_train=8, n_val=2, n_test=2),
        out_dir="runs/pilot")

ds = Dataset("runs/pilot")
for ep in ds.iter_episodes("train"):
    tr = ds.transitions(ep, mode="rgb")   # obs/action/next_obs, aligned
    targets = tr["privileged"]            # ground truth, never merged into obs
```

Headless rendering needs `QT_QPA_PLATFORM=offscreen` on this box (no display);
Panda3D itself uses EGL offscreen and does not need X.

## Layout

| path | what it is |
|---|---|
| `lewm/paths.py` | finds the sibling simulator; the only glue the split needs |
| `lewm/config.py` | `CollectConfig` — capture flags, camera, seeds, policy mixture |
| `lewm/collector.py` | the recording wrapper, `collect`, and the seed plan |
| `lewm/dataset.py` | on-disk schema, reader/writer, splits, the three views |
| `lewm/policies.py` | six driving profiles + the off-nominal injector |
| `lewm/state.py` | privileged ground truth, recorded for evaluation only |
| `lewm/validate.py` | structural / temporal / visual / action / dynamics / coverage checks |
| `lewm/pilot.py` | pilot run: collect, validate, figures, throughput, report |
| `lewm/__main__.py` | `python -m lewm collect\|pilot\|validate\|bench\|beacon\|info` |
| `tests/test_lewm.py` | compatibility, configuration, alignment, beacon, round trip, validator |
| `docs/LEWM_DATASET.md` | schema, visual setup, environment variation, compatibility report |
| `docs/lewm_pilot/` | the committed pilot: report, manifest, validation output, figures |

Later stages (pixel world model, latent probes, CEM+MPC planning) land here as
sibling packages to `lewm/`, not inside it — the dataset layer is done and
should stop changing once models depend on its schema.

## The one guarantee everything is built around

```
obs[t] + action[t] -> obs[t+1]
```

Observation-side arrays are length T+1, action-side length T, and there is no
`next_obs` on disk. It is checked two ways on every dataset: by replaying the
stored actions from the stored seed and requiring bit-identical observations,
and by re-integrating the recorded pose through the car's own kinematic model —
then doing it again with the actions shifted one step, which must fail by
orders of magnitude. A check that cannot fail is not evidence.

## Provenance

Stages 0's code was first written inside `Car-Navigation-Env` and moved here
once it was working; the reasoning for each decision is in that repository's
history (commits `5b1927c`..`ff6c5f2`) and the narrative is in `notes/journal.md`
(gitignored). Datasets record both repositories' commits in
`manifest["environment"]`, since reproducing one needs both.
