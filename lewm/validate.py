"""
Dataset validation: refuse to train on a dataset that has not passed this.

Six groups of checks, matching section 19 of the brief -- structural, temporal,
visual, action, dynamics and coverage. The two that actually earn their keep:

**Replay.** For a sample of episodes, the env is rebuilt from the manifest,
`reset(seed=episode_seed)` is called and the *stored actions* are replayed. The
observations must come back bit-identical. That single check covers temporal
alignment, determinism, seed recording and the env configuration all at once:
if any of them is wrong, the replay diverges.

**Forward-dynamics residual, with a shifted control.** For *every* episode and
every step, the recorded privileged pose is integrated one step with the car's
own kinematic model and the recorded action, and compared against the next
recorded pose. Then the same thing is computed with the actions shifted by one
step. The aligned residual must be tiny and the shifted residual must be much
larger -- which is what makes this a test rather than a statistic. A dataset
written with the classic off-by-one (`obs[t] + action[t+1] -> obs[t+1]`) passes
every structural check and fails this one by three orders of magnitude.

    python -m lewm validate runs/pilot            # all checks
    python -m lewm validate runs/pilot --replay 5 # replay more episodes

Exit status is non-zero if anything failed, so this can gate a collection run.
"""

import json
import os

import numpy as np

from .dataset import Dataset, Episode
from .config import CollectConfig

# Forward-dynamics tolerance. The pose is stored as float32 over coordinates up
# to ~190 m, so one stored metre carries ~1e-5 of representation error; a step
# at 22 m/s moves 1.1 m. 2e-3 m is ~2000x the representation error and ~2000x
# smaller than a one-step displacement, which is the gap the shifted-action
# control has to open up.
POSE_TOL_M = 2e-3
SHIFT_RATIO = 20.0      # shifted residual must exceed aligned by at least this

# Frames are compared with a tolerance, unlike every other array here, because
# the Panda3D/EGL rasteriser is not bit-reproducible across processes: two
# renders of identical geometry can disagree on a single pixel at a polygon
# edge, where the winning triangle depends on submission order. Measured on this
# box: collecting the same config twice gave 1 differing pixel in 4096 on 1 of
# 164 frames, at 7/255 per channel.
#
# The tolerance is deliberately tight on *extent* rather than on magnitude. The
# real failure this check exists to catch -- a second renderer in one process
# doubling the scene lighting -- shifts nearly every pixel a little, so it blows
# the pixel-fraction limit by three orders of magnitude while staying under any
# plausible per-channel limit.
RGB_MAX_CHANNEL_DIFF = 8
RGB_MAX_PIXEL_FRACTION = 0.001


def rgb_equivalent(a, b):
    """(ok, stats) for two frames that should be renders of the same state."""
    d = np.abs(np.asarray(a, dtype=np.int16) - np.asarray(b, dtype=np.int16))
    worst = int(d.max())
    frac = float((d.max(axis=-1) > 0).mean())
    return (worst <= RGB_MAX_CHANNEL_DIFF and frac <= RGB_MAX_PIXEL_FRACTION,
            {"max_channel_diff": worst, "pixel_fraction": frac})


class Report:
    """Accumulates pass/fail lines and prints them in the repo's test style."""

    def __init__(self, log=print):
        self.log = log
        self.rows = []

    def check(self, name, ok, detail=""):
        self.rows.append({"name": name, "ok": bool(ok), "detail": detail})
        tag = "[PASS]" if ok else "[FAIL]"
        self.log(f"  {tag} {name}" + (f"   {detail}" if detail else ""))
        return bool(ok)

    def note(self, text):
        self.log(f"         {text}")

    def section(self, title):
        self.log("")
        self.log("=" * 70)
        self.log(title)
        self.log("=" * 70)

    @property
    def failures(self):
        return [r for r in self.rows if not r["ok"]]

    @property
    def ok(self):
        return not self.failures


def _finite(a):
    return bool(np.all(np.isfinite(a)))


def _pct(x, n):
    return f"{x}/{n} = {100.0 * x / n:.1f}%" if n else "n/a"


# ----------------------------------------------------------------------
def validate_dataset(root, replay=2, log=print, strict_coverage=True):
    """Run every check against the dataset at `root`. Returns (ok, report_dict)."""
    rep = Report(log)
    ds = Dataset(root)
    cfg = CollectConfig.from_dict(ds.config)
    metas = ds.manifest["episodes"]

    rep.section(f"LeWM DATASET VALIDATION -- {root}")
    log(f"{ds.manifest['n_episodes']} episodes, "
        f"{ds.manifest['n_transitions']} transitions, "
        f"schema v{ds.manifest['schema_version']}, "
        f"mode {cfg.observation_mode}, rgb={cfg.capture_rgb}")

    stats = _structural(ds, cfg, metas, rep)
    _temporal(ds, cfg, metas, rep, replay)
    if cfg.capture_rgb:
        _visual(ds, metas, rep)
    else:
        log("")
        log("(no RGB captured -- visual checks skipped)")
    _actions(stats, rep, strict=strict_coverage)
    _dynamics(stats, rep)
    _coverage(stats, rep, strict=strict_coverage)

    rep.section("RESULT")
    if rep.ok:
        log(f"ALL {len(rep.rows)} CHECKS PASSED")
    else:
        for r in rep.failures:
            log(f"FAILED: {r['name']}  {r['detail']}")
        log(f"{len(rep.failures)} OF {len(rep.rows)} CHECK(S) FAILED")
    return rep.ok, {"ok": rep.ok, "checks": rep.rows, "stats": _summary(stats)}


# ----------------------------------------------------------------------
def _structural(ds, cfg, metas, rep):
    """File/manifest agreement, per-episode shape contract, and data gathering.

    Gathers the concatenated action/dynamics columns while it is already
    reading every episode, so the later checks cost no extra I/O.
    """
    rep.section("1. STRUCTURAL")

    root = ds.root
    disk = set()
    for dirpath, _, files in os.walk(os.path.join(root, "episodes")):
        for f in files:
            if f.endswith(".npz"):
                disk.add(os.path.relpath(os.path.join(dirpath, f), root))
    listed = {m["path"] for m in metas}
    rep.check("every manifest episode exists on disk", not (listed - disk),
              f"missing: {sorted(listed - disk)[:5]}" if listed - disk else "")
    rep.check("every episode on disk is in the manifest", not (disk - listed),
              f"unlisted: {sorted(disk - listed)[:5]}" if disk - listed else "")

    ids = [m["episode_id"] for m in metas]
    rep.check("episode ids are unique", len(set(ids)) == len(ids))

    split_ids = ds.splits
    overlaps = []
    for a in split_ids:
        for b in split_ids:
            if a < b and set(split_ids[a]) & set(split_ids[b]):
                overlaps.append((a, b))
    rep.check("train/val/test episode sets are disjoint", not overlaps,
              f"overlapping: {overlaps}" if overlaps else
              " / ".join(f"{k}={len(v)}" for k, v in sorted(split_ids.items())))

    seeds = {}
    for m in metas:
        seeds.setdefault(m["split"], set()).add(m["episode_seed"])
    all_seeds = [m["episode_seed"] for m in metas]
    rep.check("episode seeds are unique across the dataset",
              len(set(all_seeds)) == len(all_seeds))
    seed_overlap = [(a, b) for a in seeds for b in seeds if a < b and seeds[a] & seeds[b]]
    rep.check("no environment seed is shared between splits", not seed_overlap,
              f"shared: {seed_overlap}" if seed_overlap else "")

    # Per-episode contract + data gathering.
    acc = {k: [] for k in ("action", "speed", "accel", "steer", "yaw_rate",
                           "dstep", "reward", "injected", "dist_obstacle",
                           "dist_vehicle")}
    acc["lengths"] = []
    acc["reasons"] = []
    acc["policies"] = []
    acc["reached"] = []
    acc["crashed"] = []
    bad_shape, bad_time, bad_flags, nonfinite, bad_len = [], [], [], [], []
    cap = cfg.max_steps or None

    for m in metas:
        with ds.episode(m["episode_id"]) as ep:
            t = m["length"]
            acc["lengths"].append(t)
            acc["reasons"].append(m["reason"])
            acc["policies"].append(m["policy"])

            if len(ep["timestep"]) != t + 1 or len(ep["action"]) != t:
                bad_shape.append(m["episode_id"])
            if not np.array_equal(ep["timestep"], np.arange(t + 1, dtype=np.int32)):
                bad_time.append(m["episode_id"])
            limit = cap or m["max_episode_steps"]
            if not 1 <= t <= limit:
                bad_len.append((m["episode_id"], t))

            te, tr = ep["terminated"], ep["truncated"]
            # A done flag may only be set on the final step, and never both.
            flags_ok = (not np.any(te[:-1]) and not np.any(tr[:-1])
                        and not (bool(te[-1]) and bool(tr[-1])))
            if m["reason"] in ("success", "crash", "stuck") and not te[-1]:
                flags_ok = False
            if m["reason"] == "timeout" and not tr[-1]:
                flags_ok = False
            if m["reason"] == "cut" and (te[-1] or tr[-1]):
                flags_ok = False
            if not flags_ok:
                bad_flags.append(m["episode_id"])

            for name in ("obs_vector", "privileged", "action", "reward"):
                a = ep.get(name)
                if a is not None and not _finite(a):
                    nonfinite.append((m["episode_id"], name))

            acc["action"].append(np.asarray(ep["action"]))
            acc["reward"].append(np.asarray(ep["reward"]))
            acc["reached"].append(np.asarray(ep["reached"]))
            acc["crashed"].append(np.asarray(ep["crashed"]))
            if "injected" in ep:
                acc["injected"].append(np.asarray(ep["injected"]))
            priv = ep.get("privileged")
            if priv is not None:
                col = {n: i for i, n in enumerate(m["privileged_names"])}
                acc["speed"].append(priv[:, col["speed"]])
                acc["accel"].append(priv[:, col["accel"]])
                acc["steer"].append(priv[:, col["steer_angle"]])
                acc["yaw_rate"].append(priv[:, col["yaw_rate"]])
                xy = priv[:, [col["x"], col["y"]]]
                acc["dstep"].append(np.hypot(*(np.diff(xy, axis=0).T)))
                if "dist_to_obstacle" in col:
                    acc["dist_obstacle"].append(priv[:, col["dist_to_obstacle"]])
                if "dist_to_vehicle" in col:
                    acc["dist_vehicle"].append(priv[:, col["dist_to_vehicle"]])

    rep.check("observation-side records are action-side + 1", not bad_shape,
              f"offenders: {bad_shape[:5]}" if bad_shape else "")
    rep.check("timesteps are contiguous from 0", not bad_time,
              f"offenders: {bad_time[:5]}" if bad_time else "")
    rep.check("trajectory lengths are within the episode limit", not bad_len,
              f"offenders: {bad_len[:5]}" if bad_len else
              f"T: min {min(acc['lengths'])} median "
              f"{int(np.median(acc['lengths']))} max {max(acc['lengths'])}")
    rep.check("terminated/truncated only on the final step and consistent "
              "with `reason`", not bad_flags,
              f"offenders: {bad_flags[:5]}" if bad_flags else "")
    rep.check("no NaN/Inf in observations, actions, rewards or ground truth",
              not nonfinite, f"offenders: {nonfinite[:5]}" if nonfinite else "")

    required = ["timestep", "action", "reward", "terminated", "truncated"]
    if cfg.capture_vector:
        required.append("obs_vector")
    if cfg.capture_rgb:
        required.append("rgb")
    if cfg.capture_privileged_state:
        required.append("privileged")
    missing = []
    for m in metas[:1] + metas[-1:]:
        with ds.episode(m["episode_id"]) as ep:
            missing += [(m["episode_id"], r) for r in required if r not in ep]
    rep.check("every episode carries the arrays its config promises", not missing,
              f"missing: {missing}" if missing else " ".join(required))

    return {"acc": acc, "ds": ds, "cfg": cfg, "metas": metas}


# ----------------------------------------------------------------------
def _temporal(ctx_ds, cfg, metas, rep, replay):
    rep.section("2. TEMPORAL ALIGNMENT")
    ds = ctx_ds
    from env.car import Car, CarParams

    car_keys = set(CarParams.__init__.__code__.co_varnames) - {"self"}
    car_kwargs = {k: v for k, v in cfg.env_kwargs.items() if k in car_keys}
    if "car_length" in cfg.env_kwargs:
        car_kwargs["length"] = cfg.env_kwargs["car_length"]
    if "car_width" in cfg.env_kwargs:
        car_kwargs["width"] = cfg.env_kwargs["car_width"]

    aligned, shifted, n_checked, skipped = [], [], 0, []
    for m in metas:
        if m["action_repeat"] != 1:
            skipped.append(m["episode_id"])
            continue
        with ds.episode(m["episode_id"]) as ep:
            priv = ep.get("privileged")
            if priv is None:
                skipped.append(m["episode_id"])
                continue
            col = {n: i for i, n in enumerate(m["privileged_names"])}
            act = np.asarray(ep["action"], dtype=np.float64)
            dt = m["dt"]
            for shift, bucket in ((0, aligned), (1, shifted)):
                res = _forward_residual(priv, col, act, dt, shift, car_kwargs,
                                        CarParams, Car)
                if res is not None:
                    bucket.append(res)
            n_checked += 1

    if n_checked:
        a = float(np.max(aligned))
        s = float(np.max(shifted)) if shifted else float("inf")
        rep.check("obs[t] + action[t] -> obs[t+1] reproduces the car model",
                  a < POSE_TOL_M,
                  f"max pose residual {a:.2e} m over {n_checked} episodes "
                  f"(tol {POSE_TOL_M:.0e})")
        rep.check("no off-by-one: shifting the actions by one step breaks it",
                  s > SHIFT_RATIO * max(a, 1e-9),
                  f"shifted residual {s:.2e} m vs aligned {a:.2e} m "
                  f"({s / max(a, 1e-12):.0f}x)")
    else:
        rep.check("forward-dynamics residual could be computed", False,
                  "no episode had both privileged state and action_repeat=1")
    if skipped:
        rep.note(f"{len(skipped)} episode(s) skipped by the residual check")

    # --- full replay of a sample
    n = min(int(replay), len(metas))
    if n == 0:
        rep.note("replay check skipped (--replay 0)")
        return
    from .collector import build_env, build_renderer
    renderer = build_renderer(cfg)
    env = build_env(cfg, renderer)
    try:
        bad, worst_rgb, n_frames, n_px_diff = [], 0, 0, 0
        rng = np.random.default_rng(0)
        pick = rng.choice(len(metas), size=n, replace=False)
        for i in pick:
            m = metas[int(i)]
            with ds.episode(m["episode_id"]) as ep:
                obs, _ = env.reset(seed=m["episode_seed"])
                stored_vec = ep.get("obs_vector")
                stored_rgb = ep.get("rgb")
                mismatch = None
                for t, a in enumerate(np.asarray(ep["action"])):
                    if stored_vec is not None:
                        got = obs["vector"] if isinstance(obs, dict) else obs
                        if not np.array_equal(got, stored_vec[t]):
                            mismatch = f"obs_vector at t={t}"
                            break
                    if stored_rgb is not None:
                        got = obs["image"] if isinstance(obs, dict) else obs
                        same, stats = rgb_equivalent(got, stored_rgb[t])
                        n_frames += 1
                        worst_rgb = max(worst_rgb, stats["max_channel_diff"])
                        n_px_diff += stats["pixel_fraction"] > 0
                        if not same:
                            mismatch = (f"rgb at t={t} "
                                        f"({100 * stats['pixel_fraction']:.2f}% of "
                                        f"pixels, max {stats['max_channel_diff']}/255)")
                            break
                    obs, _, te, tr, _ = env.step(a)
                if mismatch:
                    bad.append((m["episode_id"], mismatch))
        detail = "bit-identical observations"
        if n_frames:
            detail = (f"vector bit-identical; {n_px_diff}/{n_frames} frames with "
                      f"any pixel difference, worst {worst_rgb}/255 "
                      f"(rasteriser tie-breaks, tolerance "
                      f"{RGB_MAX_CHANNEL_DIFF}/255 over "
                      f"{100 * RGB_MAX_PIXEL_FRACTION:.1f}% of pixels)")
        rep.check(f"replaying stored actions from the stored seed reproduces "
                  f"the episode ({n} sampled)", not bad,
                  f"diverged: {bad}" if bad else detail)
    finally:
        env.close()


def _forward_residual(priv, col, act, dt, shift, car_kwargs, CarParams, Car):
    """Max pose error from integrating the recorded action one step.

    `shift=1` is the control: it feeds `action[t+1]` into the step from `t`,
    which is the off-by-one the brief warns about, and must produce a much
    larger error than `shift=0`.
    """
    t_max = len(act) - shift
    if t_max <= 0:
        return None
    car = Car(CarParams(**car_kwargs))
    worst = 0.0
    for t in range(t_max):
        car.reset(priv[t, col["x"]], priv[t, col["y"]], priv[t, col["heading"]],
                  speed=priv[t, col["speed"]])
        car.steer_angle = float(priv[t, col["steer_angle"]])
        a = act[t + shift]
        car.step(float(a[0]), (float(a[1]) + 1.0) * 0.5, float(a[2]), dt)
        err = np.hypot(car.x - priv[t + 1, col["x"]], car.y - priv[t + 1, col["y"]])
        worst = max(worst, float(err))
    return worst


# ----------------------------------------------------------------------
def _visual(ds, metas, rep):
    rep.section("3. VISUAL")
    shapes, cams, beacons, flat, decoded = set(), set(), set(), [], 0
    for m in metas:
        with ds.episode(m["episode_id"]) as ep:
            rgb = ep["rgb"]
            shapes.add((rgb.shape[1], rgb.shape[2], rgb.shape[3], str(rgb.dtype)))
            cams.add(json.dumps(m["camera"], sort_keys=True))
            beacons.add(m["show_goal_beacon"])
            decoded += len(rgb)
            # A frame that is a single flat colour is either a lost GL context
            # or the camera inside geometry; both are corruption for our
            # purposes, and both are invisible in a shape check.
            std = rgb.reshape(len(rgb), -1).std(axis=1)
            n_flat = int((std < 1.0).sum())
            if n_flat:
                flat.append((m["episode_id"], n_flat))

    rep.check("RGB frames decode with one consistent shape and dtype",
              len(shapes) == 1, f"{sorted(shapes)}  ({decoded} frames)")
    rep.check("camera configuration is identical across the dataset",
              len(cams) == 1, f"{len(cams)} distinct configs")
    rep.check("goal-beacon setting is identical across the dataset",
              len(beacons) == 1, f"show_goal_beacon={sorted(beacons)}")
    rep.check("no flat/corrupt frames", not flat,
              f"offenders: {flat[:5]}" if flat else "")

    # Frame ordering: consecutive frames must differ while the car is moving.
    stuck = []
    for m in metas[:3]:
        with ds.episode(m["episode_id"]) as ep:
            rgb = ep["rgb"].astype(np.int16)
            priv = ep.get("privileged")
            if priv is None or len(rgb) < 3:
                continue
            col = {n: i for i, n in enumerate(m["privileged_names"])}
            moving = priv[:-1, col["speed"]] > 2.0
            same = (np.abs(np.diff(rgb, axis=0)).reshape(len(rgb) - 1, -1).max(axis=1) == 0)
            n_bad = int((moving & same).sum())
            if n_bad:
                stuck.append((m["episode_id"], n_bad))
    rep.check("frames change while the car is moving (ordering/staleness)",
              not stuck, f"offenders: {stuck[:5]}" if stuck else "")


# ----------------------------------------------------------------------
def _actions(ctx, rep, strict=True):
    rep.section("4. ACTIONS")
    acc = ctx["acc"]
    a = np.concatenate(acc["action"])
    rep.check("actions are inside the action space [-1, 1]",
              bool(np.all(np.abs(a) <= 1.0 + 1e-6)),
              f"range {a.min():.3f} .. {a.max():.3f}")
    names = ("throttle", "brake", "steer")
    std = a.std(axis=0)
    rep.check("no action channel is degenerate (constant)",
              bool(np.all(std > 1e-3)),
              "  ".join(f"{n} sd={s:.3f}" for n, s in zip(names, std)))

    # The rest of this section is about *range coverage* rather than validity:
    # the actions below are all legal, they just would not teach a world model
    # what braking does. So they are claims about a finished dataset, and
    # `strict=False` downgrades them to notes -- a 60-step smoke sample is all
    # launch phase and legitimately contains no braking at all.
    brake = (a[:, 1] + 1.0) * 0.5
    span = [
        ("steering is exercised in both directions",
         a[:, 2].min() < -0.1 and a[:, 2].max() > 0.1,
         f"steer {a[:, 2].min():.2f} .. {a[:, 2].max():.2f}"),
        ("the brake channel spans both ends of its range",
         brake.min() < 0.1 and brake.max() > 0.5,
         f"brake {brake.min():.2f} .. {brake.max():.2f} (mean {brake.mean():.2f})"),
        ("the throttle reaches full forward",
         a[:, 0].max() > 0.9,
         f"throttle {a[:, 0].min():.2f} .. {a[:, 0].max():.2f} "
         f"(reverse steps: {int((a[:, 0] < -0.05).sum())})"),
    ]
    for name, ok, detail in span:
        if strict:
            rep.check(name, bool(ok), detail)
        else:
            rep.note(f"{'ok ' if ok else 'NO '}{name}   {detail}")
    for n, lo, hi, m_ in zip(names, a.min(axis=0), a.max(axis=0), a.mean(axis=0)):
        rep.note(f"{n:<9} min {lo:+.2f}  mean {m_:+.2f}  max {hi:+.2f}")


# ----------------------------------------------------------------------
def _dynamics(ctx, rep):
    rep.section("5. DYNAMICS DISTRIBUTIONS")
    acc = ctx["acc"]
    if not acc["speed"]:
        rep.check("dynamics distributions available", False,
                  "no privileged state captured")
        return
    rows = [
        ("speed (m/s)", np.concatenate(acc["speed"])),
        ("accel (m/s2)", np.concatenate(acc["accel"])),
        ("steer angle (rad)", np.concatenate(acc["steer"])),
        ("yaw rate (rad/s)", np.concatenate(acc["yaw_rate"])),
        ("step displacement (m)", np.concatenate(acc["dstep"])),
        ("episode length (steps)", np.asarray(acc["lengths"], dtype=float)),
    ]
    rep.log(f"  {'quantity':<22}{'min':>9}{'p5':>9}{'median':>9}{'p95':>9}{'max':>9}")
    for name, v in rows:
        q = np.percentile(v, [0, 5, 50, 95, 100])
        rep.log(f"  {name:<22}" + "".join(f"{x:>9.2f}" for x in q))
    speed = np.concatenate(acc["speed"])
    rep.check("speed distribution is non-degenerate",
              float(speed.std()) > 0.5, f"sd {speed.std():.2f} m/s")
    disp = np.concatenate(acc["dstep"])
    rep.check("per-step displacement is physically plausible",
              float(disp.max()) < 22.0 * 0.05 * 1.5 + 1e-6,
              f"max {disp.max():.3f} m (limit = max_speed * dt)")


# ----------------------------------------------------------------------
COVERAGE_TARGETS = {
    # name: (predicate description, minimum fraction of steps)
    "stationary (<0.5 m/s)": 0.001,
    "low speed (0.5-3 m/s)": 0.01,
    "mid speed (3-10 m/s)": 0.10,
    "high speed (>10 m/s)": 0.02,
    "straight (|yaw rate| < 0.05)": 0.05,
    "turning (|yaw rate| > 0.2)": 0.05,
    "braking (accel < -1)": 0.02,
    "accelerating (accel > 1)": 0.02,
    "off-nominal (injected)": 0.0,
}


def _coverage(ctx, rep, strict=True):
    rep.section("6. BEHAVIOURAL COVERAGE")
    acc = ctx["acc"]
    if not acc["speed"]:
        rep.check("coverage measurable", False, "no privileged state captured")
        return
    speed = np.concatenate(acc["speed"])
    accel = np.concatenate(acc["accel"])
    yaw = np.abs(np.concatenate(acc["yaw_rate"]))
    n = len(speed)

    buckets = {
        "stationary (<0.5 m/s)": speed < 0.5,
        "low speed (0.5-3 m/s)": (speed >= 0.5) & (speed < 3.0),
        "mid speed (3-10 m/s)": (speed >= 3.0) & (speed <= 10.0),
        "high speed (>10 m/s)": speed > 10.0,
        "straight (|yaw rate| < 0.05)": yaw < 0.05,
        "turning (|yaw rate| > 0.2)": yaw > 0.2,
        "braking (accel < -1)": accel < -1.0,
        "accelerating (accel > 1)": accel > 1.0,
    }
    if acc["injected"]:
        inj = np.concatenate(acc["injected"])
        buckets["off-nominal (injected)"] = inj
        # Recovery = the 25 steps after a burst ends. Named here rather than
        # recorded because it is a window over the `injected` flag, not a
        # property of a step.
        tail = np.zeros_like(inj)
        idx = np.flatnonzero(inj[:-1] & ~inj[1:])
        for i in idx:
            tail[i + 1:i + 26] = True
        buckets["recovery (25 steps after a burst)"] = tail

    ctx["coverage"] = {}
    for name, mask in buckets.items():
        frac = float(mask.sum()) / max(1, len(mask))
        ctx["coverage"][name] = {"steps": int(mask.sum()), "total": int(len(mask)),
                                 "frac": frac, "floor": COVERAGE_TARGETS.get(name)}
        floor = COVERAGE_TARGETS.get(name)
        detail = f"{_pct(int(mask.sum()), len(mask))}"
        if floor is None or not strict:
            rep.note(f"{name:<34} {detail}")
        else:
            rep.check(f"covered: {name}", frac >= floor,
                      f"{detail}  (floor {100 * floor:.1f}%)")

    reasons = {}
    for r in acc["reasons"]:
        reasons[str(r)] = reasons.get(str(r), 0) + 1
    rep.note(f"episode endings: {reasons}")
    pol = {}
    for p in acc["policies"]:
        pol[p] = pol.get(p, 0) + 1
    rep.note(f"policy mixture: {pol}")
    reached = int(np.concatenate(acc["reached"]).sum())
    rep.note(f"waypoints reached: {reached} over {len(acc['lengths'])} episodes")
    if acc["dist_vehicle"]:
        dv = np.concatenate(acc["dist_vehicle"])
        dv = dv[dv >= 0.0]
        near = int((dv < 4.0).sum())
        rep.note(f"near-miss steps (vehicle within 4 m): {_pct(near, len(dv))}")
    if acc["dist_obstacle"]:
        do = np.concatenate(acc["dist_obstacle"])
        rep.note(f"steps with an obstacle within 3 m of the LIDAR: "
                 f"{_pct(int((do < 3.0).sum()), len(do))}")
    rep.check("the dataset contains crashes as well as clean driving",
              int(np.concatenate(acc["crashed"]).sum()) > 0
              or "crash" not in reasons,
              f"crash steps: {int(np.concatenate(acc['crashed']).sum())}")


# ----------------------------------------------------------------------
def _summary(ctx):
    acc = ctx["acc"]
    out = {
        "n_episodes": len(acc["lengths"]),
        "n_transitions": int(sum(acc["lengths"])),
        "episode_length": {
            "min": int(min(acc["lengths"])), "max": int(max(acc["lengths"])),
            "mean": float(np.mean(acc["lengths"])),
            "median": float(np.median(acc["lengths"])),
        },
        "endings": {},
        "policies": {},
        "coverage": ctx.get("coverage", {}),
    }
    for r in acc["reasons"]:
        out["endings"][str(r)] = out["endings"].get(str(r), 0) + 1
    for p in acc["policies"]:
        out["policies"][p] = out["policies"].get(p, 0) + 1
    out["waypoints_reached"] = int(np.concatenate(acc["reached"]).sum())
    out["crash_steps"] = int(np.concatenate(acc["crashed"]).sum())
    if acc["injected"]:
        out["injected_steps"] = int(np.concatenate(acc["injected"]).sum())
    if acc["speed"]:
        for name, key in (("speed", "speed"), ("accel", "accel"),
                          ("steer_angle", "steer"), ("yaw_rate", "yaw_rate")):
            v = np.concatenate(acc[key])
            out[name] = {"min": float(v.min()), "max": float(v.max()),
                         "mean": float(v.mean()), "sd": float(v.std())}
    a = np.concatenate(acc["action"])
    out["action"] = {n: {"min": float(a[:, i].min()), "max": float(a[:, i].max()),
                         "mean": float(a[:, i].mean()), "sd": float(a[:, i].std())}
                     for i, n in enumerate(("throttle", "brake", "steer"))}
    return out
