"""
The pilot run: collect a small dataset, validate it, look at it, report on it.

The brief's staging rule is that a pilot has to be inspected and signed off
before any large collection happens, so this module exists to make the
inspection cheap enough that it actually gets done:

    python -m lewm pilot ~/lewm_runs/pilot_001

which collects, validates, writes sample figures, benchmarks throughput and
leaves a `PILOT_REPORT.md` next to the data.

Three things in here are worth knowing about.

**Throughput is measured per configuration, separately** (brief section 21) --
no-recording baseline, vector-only, RGB, and RGB+privileged -- because the
whole point of the number is to decide what a full dataset costs, and the three
differ by an order of magnitude. The baseline is included so the cost of the
*instrumentation* can be separated from the cost of rendering.

**Sample frames are the recorded 64x64 pixels, upscaled with nearest-neighbour
and never resampled smoothly.** A prettier high-resolution re-render would
document a camera the model never sees. Panda3D fixes its buffer size at
`ShowBase` construction and allows one per process, so a larger render would
need its own process anyway.

**The beacon A/B figure flips `renderer.show_goal_beacon` between two captures
of the same simulator state**, so the two frames are pixel-identical except for
the waypoint posts. That is the evidence for section 11's question -- whether a
pixel model would be learning navigation or learning to read a green post --
and it only means anything if nothing else moved between the two frames.
"""

import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

from .collector import SPLITS, build_env, build_renderer, collect, collect_episode, LeWMDatasetEnv
from .config import CollectConfig, POLICY_MIX
from .dataset import Dataset
from .policies import make_policy
from .validate import validate_dataset


# The pilot's own configuration. Deliberately small: 12 episodes is enough to
# exercise every policy profile at least once (the 5% recovery share rounds to
# one episode at n=12) and to measure throughput, and small enough to throw away
# and regenerate after a change.
def pilot_config(**overrides):
    kw = dict(
        observation_mode="vector_full",
        capture_vector=True,
        capture_rgb=True,
        capture_privileged_state=True,
        show_goal_beacon=True,
        image_size=64,
        environment_seed=20261004,
        n_train=8, n_val=2, n_test=2,
        policy_mix=POLICY_MIX,
        env_kwargs={},
        note="Stage 0 pilot: instrumentation check, not a training dataset.",
    )
    kw.update(overrides)
    return CollectConfig(**kw)


# ----------------------------------------------------------------------
# Throughput (brief section 21)
# ----------------------------------------------------------------------
_NO_RGB = dict(capture_vector=True, capture_rgb=False,
               capture_privileged_state=False)
_RGB = dict(capture_vector=True, capture_rgb=True,
            capture_privileged_state=False)
_RGB_PRIV = dict(capture_vector=True, capture_rgb=True,
                 capture_privileged_state=True)

# (label, capture flags, record?, run the scripted policy?)
# Read as a cumulative breakdown: each row adds one cost to the row above, so
# the differences are attributable. The first row is the simulator alone, which
# is the number README.md quotes; the second shows what the scripted driver
# costs, which turns out to dominate vector-only collection.
BENCH_CASES = (
    ("env.step only (constant action)", _NO_RGB, False, False),
    ("+ scripted policy", _NO_RGB, False, True),
    ("+ vector recording", _NO_RGB, True, True),
    ("+ RGB capture", _RGB, True, True),
    ("+ privileged state", _RGB_PRIV, True, True),
)


def benchmark(base_config=None, n_episodes=2, max_steps=250, log=print):
    """Steps/s for each capture configuration, measured one at a time.

    Everything except the capture flags is held fixed, including the episode
    seeds and the policy, so the differences between rows are the cost of
    recording and rendering rather than of driving somewhere else.

    `env.reset` is excluded from every row and reported separately. It is the
    expensive call -- procedural city generation plus a scene-graph rebuild --
    and including it would make the steps/s figure depend on episode length,
    which is a property of the policy mixture rather than of the capture
    configuration being measured. `collect_episode` already starts its clock
    after the reset for the same reason.
    """
    base = base_config or pilot_config()
    rows = []
    for name, flags, record, use_policy in BENCH_CASES:
        cfg = CollectConfig.from_dict({**base.to_dict(), **flags})
        renderer = build_renderer(cfg)
        env = build_env(cfg, renderer)
        rec = LeWMDatasetEnv(env, cfg) if record else None
        try:
            # One untimed episode first: the renderer builds its scene graph and
            # numpy warms up on the first reset, and charging that to the
            # measurement makes a short benchmark look 2x slower than reality.
            _warmup(env, rec, cfg, max_steps=40)
            steps, bytes_, wall, resets, reset_wall = 0, 0, 0.0, 0, 0.0
            for i in range(n_episodes):
                seed = 900_000 + i
                if rec is not None:
                    # One extra reset purely to time it; `collect_episode` does
                    # its own and starts its clock afterwards.
                    t0 = time.perf_counter()
                    rec.reset(seed=seed)
                    reset_wall += time.perf_counter() - t0
                    resets += 1
                    arrays, stats = collect_episode(
                        rec, make_policy("competent", env, np.random.default_rng(i)),
                        seed, max_steps=max_steps)
                    steps += stats["length"]
                    wall += stats["wall_s"]
                    bytes_ += sum(a.nbytes for a in arrays.values())
                else:
                    n, dt_steps, dt_reset = _drive(env, seed, max_steps, use_policy)
                    steps += n
                    wall += dt_steps
                    reset_wall += dt_reset
                    resets += 1
        finally:
            env.close()
        rows.append({
            "case": name, "steps": steps, "wall_s": wall,
            "steps_per_s": steps / wall if wall else float("nan"),
            "bytes_per_step": bytes_ / steps if (bytes_ and steps) else None,
            "reset_ms": 1000.0 * reset_wall / resets if resets else None,
        })
        log(f"  {name:<34}{rows[-1]['steps_per_s']:>8.0f} steps/s"
            f"   reset {rows[-1]['reset_ms']:>5.0f} ms"
            + (f"   {rows[-1]['bytes_per_step'] / 1024:>7.1f} KiB/step"
               if rows[-1]["bytes_per_step"] else ""))
    return rows


def _warmup(env, rec, cfg, max_steps):
    target = rec if rec is not None else env
    obs, info = target.reset(seed=12345)
    policy = make_policy("competent", env, np.random.default_rng(0))
    for _ in range(max_steps):
        obs, _, te, tr, info = target.step(policy.act(obs, info))
        if te or tr:
            break


def _drive(env, seed, max_steps, use_policy):
    """Run the bare env with no recording, with or without the scripted driver.

    Termination is ignored rather than reset on, so that every row costs the
    same one reset per episode. The no-policy variant holds a constant
    part-throttle action and so crashes early; resetting each time it did would
    make the row a measurement of city generation instead of `env.step`.
    """
    const = np.array([0.5, -1.0, 0.0], dtype=np.float32)
    t0 = time.perf_counter()
    obs, info = env.reset(seed=seed)
    dt_reset = time.perf_counter() - t0
    policy = make_policy("competent", env, np.random.default_rng(0)) if use_policy else None
    t0 = time.perf_counter()
    for _ in range(max_steps):
        action = policy.act(obs, info) if policy is not None else const
        obs, _, _, _, info = env.step(action)
    return max_steps, time.perf_counter() - t0, dt_reset


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def _tile(frames, scale=3, cols=None, pad=2, bg=(24, 24, 28)):
    """Tile uint8 frames into one image. Nearest-neighbour upscale only."""
    from PIL import Image
    n = len(frames)
    cols = cols or n
    rows = int(np.ceil(n / cols))
    h, w = frames[0].shape[:2]
    H, W = h * scale, w * scale
    out = Image.new("RGB", (cols * W + (cols + 1) * pad,
                            rows * H + (rows + 1) * pad), bg)
    for i, f in enumerate(frames):
        r, c = divmod(i, cols)
        img = Image.fromarray(np.asarray(f, dtype=np.uint8))
        out.paste(img.resize((W, H), Image.NEAREST),
                  (pad + c * (W + pad), pad + r * (H + pad)))
    return out


def sample_frames(root, out_dir=None, n_episodes=4, n_frames=8, scale=3, log=print):
    """A strip of evenly spaced frames per sampled episode. Returns file paths."""
    ds = Dataset(root)
    out_dir = out_dir or os.path.join(root, "samples")
    os.makedirs(out_dir, exist_ok=True)
    ids = ds.episode_ids()
    pick = ids[:: max(1, len(ids) // max(1, n_episodes))][:n_episodes]
    written = []
    for eid in pick:
        with ds.episode(eid) as ep:
            if "rgb" not in ep:
                return written
            rgb = ep["rgb"]
            idx = np.linspace(0, len(rgb) - 1, min(n_frames, len(rgb))).astype(int)
            img = _tile([rgb[i] for i in idx], scale=scale)
            name = f"frames_{eid.replace('/', '_')}.png"
            path = os.path.join(out_dir, name)
            img.save(path)
            written.append(path)
            log(f"  wrote {path}  ({ep.meta['policy']}, steps {idx.tolist()})")
    return written


def path_map(root, out_path=None, size=760, margin=20, log=print):
    """Every episode's driven path in one image, coloured by split.

    The cheapest honest answer to "does the pilot cover the map or does it
    drive the same street twelve times", which is a coverage question the
    per-step histograms in `validate` cannot answer.
    """
    from PIL import Image, ImageDraw
    ds = Dataset(root)
    colours = {"train": (90, 200, 255), "val": (255, 190, 80), "test": (255, 110, 160)}
    paths = []
    for m in ds.manifest["episodes"]:
        with ds.episode(m["episode_id"]) as ep:
            priv = ep.get("privileged")
            if priv is None:
                return None
            col = {n: i for i, n in enumerate(m["privileged_names"])}
            paths.append((m["split"], priv[:, [col["x"], col["y"]]],
                          np.asarray(m["targets"], dtype=float)))

    # The extent is the city, not the paths: a plot auto-scaled to the paths
    # would hide exactly the finding we are looking for.
    env_kwargs = ds.config["env_kwargs"]
    tile = float(env_kwargs.get("tile_size", 4.0))
    extent = (max(float(env_kwargs.get("width", 64)),
                  float(env_kwargs.get("height", 64))) * tile)
    s = (size - 2 * margin) / extent

    img = Image.new("RGB", (size, size), (18, 18, 22))
    d = ImageDraw.Draw(img)
    for g in range(0, int(extent) + 1, int(10 * tile)):
        p = margin + g * s
        d.line([(p, margin), (p, size - margin)], fill=(32, 32, 38))
        d.line([(margin, p), (size - margin, p)], fill=(32, 32, 38))

    def xy(p):
        return (margin + p[0] * s, size - margin - p[1] * s)   # y up

    for split, pts, targets in paths:
        d.line([xy(p) for p in pts], fill=colours.get(split, (200, 200, 200)), width=2)
        for t in targets:
            x, y = xy(t)
            d.ellipse([x - 3, y - 3, x + 3, y + 3], outline=(120, 255, 150))
        x, y = xy(pts[0])
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 255, 255))

    out_path = out_path or os.path.join(root, "samples", "paths.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)
    log(f"  wrote {out_path}  ({len(paths)} paths, {extent:.0f} m extent)")
    return out_path


def beacon_figure(out_path, seed=7, n_frames=5, image_size=192, max_steps=400,
                  episodes=2, scale=1, log=print):
    """Beacon on/off A/B, plus how often the beacon is visible at all.

    Both frames of a column come from the *same* simulator state: the flag is
    flipped between two `capture()` calls, so the only difference within a
    column is the posts. Needs a process of its own, because the Panda3D buffer
    size is fixed when the first renderer in the process is built -- `run_pilot`
    re-invokes `python -m lewm beacon` for exactly that reason.

    Every step is captured twice and the *most affected* frames are the ones
    shown, because sampling fixed steps produces a figure of five identical
    pairs: waypoints are sampled 40-110 m from the previous one (so the chain
    reaches ~180 m from the start) and the city's buildings are 6-24 m tall,
    so the posts are occluded for most of an episode and the
    honest summary is a visibility rate rather than a picture. That rate is the
    actual answer to section 11 -- a shortcut that is invisible 90% of the time
    is a different problem from one the camera can always see.
    """
    cfg = pilot_config(capture_rgb=True, image_size=image_size)
    renderer = build_renderer(cfg)
    env = build_env(cfg, renderer)
    try:
        frames, fracs = [], []
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            policy = make_policy("competent", env, np.random.default_rng(seed))
            for t in range(max_steps):
                renderer.show_goal_beacon = True
                on = renderer.capture(env)
                renderer.show_goal_beacon = False
                off = renderer.capture(env)
                d = np.abs(on.astype(np.int16) - off.astype(np.int16)).sum(axis=2)
                frac = float((d > 8).mean())
                fracs.append(frac)
                if frac > 0.0:
                    frames.append((frac, on, off))
                    # Keep only the best handful; an episode is hundreds of
                    # 192x192 pairs and all but a few are thrown away.
                    frames.sort(key=lambda r: -r[0])
                    del frames[n_frames:]
                renderer.show_goal_beacon = True
                obs, _, te, tr, info = env.step(policy.act(obs, info))
                if te or tr:
                    break
        visible = [f for f in fracs if f > 0.0]
        info_out = {
            "path": out_path,
            "steps_measured": len(fracs),
            "visible_fraction_of_steps": len(visible) / max(1, len(fracs)),
            "changed_pixel_fraction_when_visible": {
                "mean": float(np.mean(visible)) if visible else 0.0,
                "max": float(np.max(visible)) if visible else 0.0,
            },
            "frames_shown": [float(f) for f, _, _ in frames],
        }
        if not frames:
            log(f"  beacon never visible over {len(fracs)} steps; no figure written")
            return info_out
        # Chronological order was lost by the sort; brightest-first is the more
        # useful ordering for a figure meant to show the maximum exposure.
        img = _tile([f[1] for f in frames] + [f[2] for f in frames],
                    scale=scale, cols=len(frames))
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        img.save(img_path := out_path)
        log(f"  wrote {img_path}  beacon visible in "
            f"{100 * info_out['visible_fraction_of_steps']:.1f}% of "
            f"{len(fracs)} steps; when visible it covers "
            f"{100 * info_out['changed_pixel_fraction_when_visible']['mean']:.2f}% "
            f"of pixels (max "
            f"{100 * info_out['changed_pixel_fraction_when_visible']['max']:.2f}%)")
        return info_out
    finally:
        env.close()


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
def _cell(x):
    # Check names carry things like "|yaw rate| < 0.05", which would otherwise
    # split a markdown row into extra columns.
    return str(x).replace("|", r"\|")


def _table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_cell(c) for c in r) + " |")
    return "\n".join(out)


def write_report(root, report, bench=None, figures=None, beacon=None):
    """Write `PILOT_REPORT.md` from the validation report and the benchmark."""
    ds = Dataset(root)
    man = ds.manifest
    cfg = man["config"]
    st = report["stats"]
    L = []
    w = L.append

    w(f"# LeWM pilot dataset report\n")
    w(f"- **dataset**: `{os.path.abspath(root)}`")
    envinfo = man.get("environment") or {}
    w(f"- **collected**: {man.get('collected_at')}  "
      f"(lewm `{(envinfo.get('git_sha') or '?')[:10]}`, "
      f"simulator `{(envinfo.get('carnav_git_sha') or '?')[:10]}`)")
    w(f"- **schema version**: {man['schema_version']}")
    w(f"- **episodes**: {man['n_episodes']} "
      f"({'/'.join(f'{k} {len(v)}' for k, v in sorted(man['splits'].items()))})")
    w(f"- **transitions**: {man['n_transitions']}")
    w(f"- **observation mode**: `{cfg['observation_mode']}`; "
      f"captured: vector={cfg['capture_vector']}, rgb={cfg['capture_rgb']}, "
      f"privileged={cfg['capture_privileged_state']}")
    w(f"- **validation**: {'PASSED' if report['ok'] else 'FAILED'} "
      f"({sum(1 for c in report['checks'] if c['ok'])}/{len(report['checks'])} checks)")
    w(f"\n> {cfg.get('note', '')}\n")

    w("## 1. Status against the brief\n")
    w(f"Stage 0 only. This is an instrumentation check, **not** a training "
      f"dataset: {man['n_episodes']} episodes and {man['n_transitions']} "
      f"transitions are orders of magnitude short of what the brief's stage 1 "
      f"asks for, and no model has been trained on it. What it establishes is "
      f"that the recorded trajectories are correct -- aligned, reproducible, "
      f"complete and separated from the privileged state -- so that scaling up "
      f"is a matter of changing three episode counts.\n")

    w("## 2. Validation\n")
    w(_table(["", "check", "detail"],
             [("PASS" if c["ok"] else "**FAIL**", c["name"], c["detail"] or "")
              for c in report["checks"]]))
    w("")
    w("The two checks that carry the weight are the replay check -- re-running "
      "the stored actions from the stored seed must reproduce the stored "
      "observations bit-for-bit -- and the forward-dynamics residual, which "
      "integrates the recorded pose one step with the recorded action and the "
      "car's own kinematic model, then does it again with the actions shifted "
      "by one step. The aligned residual is at float32 noise and the shifted "
      "one is thousands of times larger, which is what makes "
      "`obs[t] + action[t] -> obs[t+1]` a measured property rather than a "
      "claim in a docstring.\n")

    w("## 3. Dataset composition\n")
    w(_table(["policy profile", "target share", "episodes"],
             [(k, f"{100 * cfg['policy_mix'].get(k, 0.0):.0f}%", v)
              for k, v in sorted(st["policies"].items())]))
    w("")
    w(_table(["episode ending", "count"], sorted(st["endings"].items())))
    w("")
    el = st["episode_length"]
    w(f"Episode length: min {el['min']}, median {el['median']:.0f}, "
      f"mean {el['mean']:.0f}, max {el['max']} steps "
      f"({el['mean'] * man['episodes'][0]['dt']:.0f} s of simulated driving on "
      f"average).")
    w(f"Waypoints reached: {st.get('waypoints_reached')}; "
      f"crash steps: {st.get('crash_steps')}; "
      f"injected off-nominal steps: {st.get('injected_steps')}.\n")

    if st.get("coverage"):
        w("## 4. Behavioural coverage\n")
        w(_table(["regime", "steps", "share", "floor"],
                 [(k, v["steps"], f"{100 * v['frac']:.1f}%",
                   "-" if v["floor"] is None else f"{100 * v['floor']:.1f}%")
                  for k, v in st["coverage"].items()]))
        w("")

    w("## 5. Dynamics\n")
    rows = [(k, f"{st[k]['min']:.2f}", f"{st[k]['mean']:.2f}",
             f"{st[k]['max']:.2f}", f"{st[k]['sd']:.2f}")
            for k in ("speed", "accel", "steer_angle", "yaw_rate") if k in st]
    w(_table(["quantity", "min", "mean", "max", "sd"], rows))
    w("")
    w(_table(["action channel", "min", "mean", "max", "sd"],
             [(k, f"{v['min']:.2f}", f"{v['mean']:.2f}", f"{v['max']:.2f}",
               f"{v['sd']:.2f}") for k, v in st["action"].items()]))
    w("")

    if bench:
        w("## 6. Throughput\n")
        w(_table(["configuration", "steps/s", "env.reset (ms)", "KiB/step",
                  "steps measured"],
                 [(r["case"], f"{r['steps_per_s']:.0f}",
                   "-" if r.get("reset_ms") is None else f"{r['reset_ms']:.0f}",
                   "-" if not r["bytes_per_step"] else f"{r['bytes_per_step'] / 1024:.1f}",
                   r["steps"]) for r in bench]))
        w("")
        w("Cumulative: each row adds one cost to the row above it, so the "
          "differences are attributable.\n")
        vec = next((r for r in bench if r["case"] == "+ vector recording"), None)
        rgb = next((r for r in bench if r["case"] == "+ privileged state"), None)
        raw = next((r for r in bench if r["case"].startswith("env.step")), None)
        pol = next((r for r in bench if r["case"] == "+ scripted policy"), None)
        if raw and pol:
            w(f"The scripted driver, not the simulator, is what a vector-only "
              f"collection spends its time on: {raw['steps_per_s']:.0f} steps/s "
              f"for `env.step` alone against {pol['steps_per_s']:.0f} with the "
              f"controller in the loop. Recording on top of that is nearly free.\n")
        if vec and rgb:
            w(f"RGB capture costs {vec['steps_per_s'] / rgb['steps_per_s']:.1f}x in "
              f"wall-clock. At the measured {rgb['steps_per_s']:.0f} steps/s, one "
              f"million RGB transitions is about "
              f"{1e6 / rgb['steps_per_s'] / 3600:.1f} h in a single process and "
              f"{1e6 * rgb['bytes_per_step'] / 1e9:.0f} GB on disk uncompressed -- "
              f"the number that decides the size of the real collection, and the "
              f"reason the seed plan was written to be shardable by split and "
              f"episode index (each episode's seeds are a pure function of "
              f"`(environment_seed, split, index)`, so N processes can each take "
              f"a stride without coordinating).\n")
        tp = man.get("throughput") or {}
        if tp.get("bytes_per_transition"):
            w(f"The pilot itself wrote {tp['bytes_on_disk'] / 1e6:.1f} MB "
              f"({tp['bytes_per_transition'] / 1024:.1f} KiB/transition) in "
              f"{tp['wall_s']:.0f} s.\n")

    if figures:
        w("## 7. Sample frames\n")
        w("What the pixel view actually contains, upscaled nearest-neighbour "
          "from the recorded 64x64 frames -- not re-rendered at a higher "
          "resolution, because the point is to show what a model would be "
          "trained on.\n")
        for p in figures:
            w(f"![{os.path.basename(p)}]({os.path.relpath(p, root)})")
        w("")

    if beacon:
        w("## 8. Goal beacon (brief section 11)\n")
        vis = beacon.get("visible_fraction_of_steps", 0.0)
        cov = beacon.get("changed_pixel_fraction_when_visible") or {}
        w(f"Measured by capturing every step twice, flipping "
          f"`show_goal_beacon` between the two captures of the *same* simulator "
          f"state, over {beacon.get('steps_measured', 0)} steps of competent "
          f"driving:\n")
        w(f"- the beacon changes at least one pixel in "
          f"**{100 * vis:.1f}%** of steps")
        w(f"- when visible it covers **{100 * cov.get('mean', 0.0):.2f}%** of the "
          f"frame on average, at most {100 * cov.get('max', 0.0):.2f}%\n")
        if os.path.exists(beacon["path"]):
            w(f"Top row: `show_goal_beacon=True` (the default, and what every "
              f"existing checkpoint and screenshot was produced with). Bottom "
              f"row: the same states with the posts hidden. The columns are the "
              f"frames where the beacon is *most* visible, not a fixed sample -- "
              f"a fixed sample produces five identical-looking pairs, which is "
              f"itself the finding.\n")
            w(f"![beacon]({os.path.relpath(beacon['path'], root)})")
            w("")
        w(f"Interpretation: waypoints are sampled 40-110 m from the previous one "
          f"(`target_min_dist`/`target_max_dist`, so the chain reaches ~180 m "
          f"from the spawn) and the city's buildings are 6-24 m tall, so the posts are "
          f"occluded for most of a run and only appear once the car is on the "
          f"right street. So the beacon is not a free answer to the whole "
          f"navigation problem -- but it is a strong local cue exactly when the "
          f"goal is nearly reached, which is the part of the task a pixel world "
          f"model would most plausibly shortcut. That is an argument for "
          f"collecting the beacon-off control (`--no-beacon`), not a reason to "
          f"skip it: the switch exists and the two datasets differ in nothing "
          f"else.\n")

    w("## 9. What this pilot does not establish\n")
    w("- Nothing has been trained. Dataset *correctness* is checked; dataset "
      "*sufficiency* for learning a world model is not, and cannot be without "
      "a training run.\n"
      "- The city is procedural but the map distribution is fixed by "
      "`CityConfig`; this pilot varies the seed, not the generator.\n"
      "- 12 episodes cannot establish the tail: rare events (red-light "
      "violations, pedestrian conflicts, reverse manoeuvres) appear a handful "
      "of times at most, so their frequency here says nothing about their "
      "frequency at scale.\n")
    return _write(os.path.join(root, "PILOT_REPORT.md"), "\n".join(L) + "\n")


def _write(path, text):
    with open(path, "w") as f:
        f.write(text)
    return path


# ----------------------------------------------------------------------
def run_pilot(config=None, out_dir="runs/lewm_pilot", bench=True, beacon=True,
              replay=3, log=print):
    """Collect, validate, illustrate and report. The whole stage-0 deliverable."""
    config = config or pilot_config()
    log("")
    log("=" * 70)
    log(f"COLLECTING PILOT -> {out_dir}")
    log("=" * 70)
    collect(config, out_dir, log=log)

    ok, report = validate_dataset(out_dir, replay=replay, log=log)

    log("")
    log("=" * 70)
    log("FIGURES")
    log("=" * 70)
    figures = sample_frames(out_dir, log=log)
    p = path_map(out_dir, log=log)
    if p:
        figures.append(p)

    beacon_info = None
    if beacon and config.capture_rgb:
        # Own process: the Panda3D buffer size is fixed by the first renderer
        # built in a process, and this figure wants a larger one than 64.
        out = os.path.join(out_dir, "samples", "beacon_ab.png")
        proc = subprocess.run([sys.executable, "-m", "lewm", "beacon", out],
                              capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        log(proc.stdout.strip() or proc.stderr.strip()[-400:])
        meta = os.path.splitext(out)[0] + ".json"
        if os.path.exists(meta):
            with open(meta) as f:
                beacon_info = json.load(f)

    bench_rows = None
    if bench:
        log("")
        log("=" * 70)
        log("THROUGHPUT (measured per configuration)")
        log("=" * 70)
        bench_rows = benchmark(config, log=log)
        report["benchmark"] = bench_rows

    path = write_report(out_dir, report, bench=bench_rows, figures=figures,
                        beacon=beacon_info)
    with open(os.path.join(out_dir, "validation.json"), "w") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=float)
    log("")
    log(f"wrote {path}")
    log(f"validation: {'PASSED' if ok else 'FAILED'}")
    return ok, report
