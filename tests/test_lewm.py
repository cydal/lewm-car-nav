"""
Checks for the LeWM dataset instrumentation, and for the environment it wraps.

Two halves, in this order on purpose. The first half is the brief's section 28
compatibility checklist: the existing environment, controllers and renderer must
behave exactly as they did before `lewm/` existed. The strongest form of that is
not "it still runs" but *bit-identical rollouts* -- the same seed and the same
action sequence through the bare env and through the recording wrapper must
produce the same observations, which is only true if recording reads state and
never touches it (no extra RNG draws, no re-scanning the LIDAR).

The second half checks the instrumentation itself, and the check that matters
most is section 13's: `observation[t] + action[t] -> observation[t+1]`, with a
*deterministic* action sequence rather than a policy, verified by re-integrating
the car model -- and then verified again with the actions shifted by one step,
which must fail. A test for an off-by-one that cannot detect an off-by-one is
not a test. The last section goes further and deliberately corrupts a dataset on
disk to confirm `validate_dataset` rejects it.

Needs panda3d (for the RGB sections) and numpy. Run it as a script:

    python tests/test_lewm.py
"""

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# `lewm` first: importing it puts the sibling Car-Navigation-Env checkout on
# sys.path (lewm/paths.py), which is what makes the three simulator imports
# below resolve. Half of this file tests the *env*, so those imports are the
# point rather than an implementation detail.
from lewm import (CAMERA_DEFAULTS, CollectConfig, Dataset, LeWMDatasetEnv,
                  POLICY_MIX, PROFILES, PrivilegedRecorder, carnav_root,
                  collect, make_policy, policy_schedule, validate_dataset)

import carnav
from baselines.scripted import GapFollower
from env.car import Car, CarParams
from lewm.collector import _seed_plan, build_env, build_renderer
from lewm.dataset import transitions, write_episode
from lewm.validate import rgb_equivalent

TMP = tempfile.mkdtemp(prefix="lewm_test_")
IMAGE_SIZE = 64          # one buffer size per process: Panda3D fixes it at
                         # ShowBase construction, so every RGB section here
                         # must use the same one.
CHECKS = [0]


def ok(label, detail=""):
    CHECKS[0] += 1
    print(f"  [OK] {label}" + (f"   {detail}" if detail else ""))


def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def fixed_actions(n, seed=0):
    """A deterministic action sequence with no policy in the loop.

    Built from a fixed RNG rather than a constant so that it exercises the
    steering actuator's rate limit and both ends of the brake channel, while
    still being exactly reproducible -- the point of section 13's test is that
    the alignment holds for *arbitrary* actions, not just for ones a controller
    would choose.
    """
    rng = np.random.default_rng(seed)
    a = np.stack([
        rng.uniform(-0.2, 1.0, n),       # throttle, mostly forward
        rng.uniform(-1.0, 0.4, n),       # brake, mostly released
        rng.uniform(-1.0, 1.0, n),       # steer
    ], axis=1).astype(np.float32)
    a[n // 3:n // 3 + 10] = [0.0, 1.0, 0.0]     # a sustained full-brake burst
    return a


# ======================================================================
section("1. COMPATIBILITY: the existing environment is unchanged")
# ======================================================================
print(f"  simulator: {carnav_root() or os.path.dirname(carnav.__file__)}")
env = carnav.make(obs_type="vector")
obs, info = env.reset(seed=3)
assert obs.shape == (env.vector_dim,) and obs.dtype == np.float32
assert np.all(np.isfinite(obs)) and obs.min() >= -1.0 - 1e-5 and obs.max() <= 1.0 + 1e-5
for _ in range(200):
    obs, r, te, tr, info = env.step(env.action_space.sample())
    if te or tr:
        obs, info = env.reset()
ok("default env resets and steps, observation stays in its box",
   f"dim {env.vector_dim}")

driver = GapFollower.for_env(env)
obs, info = env.reset(seed=4)
driver.reset()
for _ in range(300):
    obs, r, te, tr, info = env.step(driver.act(obs, info))
    if te or tr:
        break
assert set(driver.diagnostics())
ok("the existing scripted controller still drives it",
   f"reward {info['episode_reward']:.1f}, reason {info.get('reason')}")

e2 = carnav.make(obs_type="vector", traffic=False, width=32, height=32)
o2, _ = e2.reset(seed=1)
e2.step(np.zeros(3, dtype=np.float32))
assert o2.shape[0] < env.vector_dim, "traffic=False should shrink the observation"
e2.close()
try:
    carnav.make(obs_type="vector", nonsense_keyword=1)
except TypeError:
    ok("carnav.make still routes flat keywords and still rejects unknown ones",
       f"traffic=False -> {o2.shape[0]}D instead of {env.vector_dim}D")
else:
    raise AssertionError("carnav.make accepted an unknown keyword")

# The instrumentation must be transparent: identical seed, identical actions,
# identical observations. Anything the recorder did to the env -- an extra draw
# on an RNG stream, a second LIDAR scan, a mutated car -- would show up here.
ACTS = fixed_actions(120, seed=11)
bare = carnav.make(obs_type="vector")
o0, _ = bare.reset(seed=77)
bare_obs = [o0.copy()]
bare_rew = []
for a in ACTS:
    o, r, te, tr, _ = bare.step(a)
    bare_obs.append(o.copy())
    bare_rew.append(r)
    if te or tr:
        break
bare.close()

wrapped_env = carnav.make(obs_type="vector")
rec = LeWMDatasetEnv(wrapped_env, CollectConfig(capture_rgb=False))
o0, _ = rec.reset(seed=77)
wrapped_obs = [o0.copy()]
wrapped_rew = []
for a in ACTS[:len(bare_rew)]:
    o, r, te, tr, _ = rec.step(a)
    wrapped_obs.append(o.copy())
    wrapped_rew.append(r)
    if te or tr:
        break
arrays = rec.arrays()
wrapped_env.close()

assert len(bare_obs) == len(wrapped_obs)
for t, (a, b) in enumerate(zip(bare_obs, wrapped_obs)):
    assert np.array_equal(a, b), f"observation {t} differs with recording on"
assert np.allclose(bare_rew, wrapped_rew, rtol=0, atol=0)
ok("recording is side-effect free: bit-identical rollout with and without it",
   f"{len(bare_rew)} steps, {len(bare_obs)} observations")

assert np.array_equal(arrays["obs_vector"], np.stack(wrapped_obs))
assert np.array_equal(arrays["action"], ACTS[:len(bare_rew)])
ok("what was recorded is exactly what was observed and commanded")

# ---- renderer: defaults unchanged, and CAMERA_DEFAULTS is not a stale copy
from render.panda_renderer import PandaRenderer
import inspect
sig = inspect.signature(PandaRenderer.__init__)
for key, val in CAMERA_DEFAULTS.items():
    actual = sig.parameters[key].default
    assert actual == val, f"CAMERA_DEFAULTS[{key}]={val} but renderer default is {actual}"
assert sig.parameters["show_goal_beacon"].default is True, \
    "the goal beacon must stay on by default -- every existing checkpoint and " \
    "screenshot was produced with it visible"
ok("CAMERA_DEFAULTS matches the renderer's real defaults, beacon defaults on",
   ", ".join(f"{k}={v}" for k, v in sorted(CAMERA_DEFAULTS.items())))

rgb_cfg = CollectConfig(capture_rgb=True, image_size=IMAGE_SIZE)
renderer = build_renderer(rgb_cfg)
img_env = build_env(rgb_cfg, renderer)
obs, info = img_env.reset(seed=5)
assert set(obs) == {"vector", "image"}
assert obs["image"].shape == (IMAGE_SIZE, IMAGE_SIZE, 3) and obs["image"].dtype == np.uint8
assert obs["image"].std() > 1.0
ok("image observations still render headless",
   f"{obs['image'].shape} range {obs['image'].min()}..{obs['image'].max()}")

# ======================================================================
section("2. CONFIGURATION")
# ======================================================================
for bad, why in (
    (dict(observation_mode="pixels"), "unknown observation mode"),
    (dict(policy_mix={"competent": 0.5}), "policy shares that do not sum to 1"),
    (dict(camera={"zoom": 2.0}), "unknown camera setting"),
    (dict(observation_mode="rgb", capture_rgb=False), "rgb mode without rgb capture"),
    (dict(capture_vector=False, capture_rgb=False), "capturing nothing"),
):
    try:
        CollectConfig(**bad)
    except ValueError:
        continue
    raise AssertionError(f"CollectConfig accepted {why}")
ok("invalid configurations raise instead of collecting the wrong thing")

cfg = CollectConfig(capture_rgb=True, n_train=3, note="round trip",
                    env_kwargs={"traffic": False})
path = os.path.join(TMP, "cfg.json")
cfg.save(path)
back = CollectConfig.load(path)
assert back.to_dict() == cfg.to_dict()
assert CollectConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()
ok("config round-trips through JSON unchanged")

assert CollectConfig(capture_rgb=False).obs_type == "vector"
assert CollectConfig(capture_rgb=True).obs_type == "both"
assert CollectConfig(capture_rgb=True, capture_vector=False,
                     observation_mode="rgb").obs_type == "image"
ok("capture flags map to the env's obs_type",
   "rgb+vector -> 'both', so one _observe() produces both and they cannot drift")

assert abs(sum(POLICY_MIX.values()) - 1.0) < 1e-12
sched = policy_schedule(100, POLICY_MIX, np.random.default_rng(0))
counts = {k: sched.count(k) for k in POLICY_MIX}
assert counts == {k: int(round(100 * v)) for k, v in POLICY_MIX.items()}, counts
assert policy_schedule(12, POLICY_MIX, np.random.default_rng(1)).count("recovery") == 1
ok("the policy schedule hits the advertised composition exactly", str(counts))

# The mixture is apportioned over the dataset and dealt to the splits, not
# apportioned per split: at 8/2/2 the per-split rounding drops the 5% recovery
# share to zero in all three splits, so a pilot would contain none of the
# off-nominal bursts while its manifest still claimed 5%.
plan = _seed_plan(CollectConfig(n_train=8, n_val=2, n_test=2))
dealt = [row["policy"] for split in ("train", "val", "test") for row in plan[split]]
assert len(dealt) == 12
assert dealt.count("recovery") == 1, dealt
assert {k: dealt.count(k) for k in POLICY_MIX} == \
       {"competent": 3, "noisy_competent": 2, "aggressive": 3,
        "intermediate": 1, "conservative": 2, "recovery": 1}, dealt
ok("a 12-episode dataset still gets the whole mixture, recovery included",
   f"{len(set(dealt))}/6 profiles present across 8/2/2 splits")

# ======================================================================
section("3. PRIVILEGED STATE IS SEPARATE AND TRUE")
# ======================================================================
vec_env = carnav.make(obs_type="vector")
pr = PrivilegedRecorder(vec_env)
vec_env.reset(seed=9)
names = list(pr.names)
assert len(set(names)) == len(names)
row = pr.row()
assert row.shape == (len(names),) and np.all(np.isfinite(row))
col = {n: i for i, n in enumerate(names)}
for key in ("x", "y", "heading", "speed", "accel", "dist_to_goal",
            "dist_to_obstacle"):
    assert key in col, f"missing privileged column {key}"
ok("privileged columns are present, unique and finite", f"{len(names)} columns")

for _ in range(40):
    vec_env.step(np.array([0.8, -1.0, 0.1], dtype=np.float32))
row = pr.row()
car = vec_env.car
# Tolerances are float32 storage error over coordinates up to ~250 m, not
# modelling slack: the recorder copies these values, it does not compute them.
assert abs(row[col["x"]] - car.x) < 1e-3 and abs(row[col["y"]] - car.y) < 1e-3
assert abs(row[col["speed"]] - car.speed) < 1e-4
gx, gy = vec_env.targets[vec_env.target_idx]
assert abs(row[col["dist_to_goal"]] - np.hypot(gx - car.x, gy - car.y)) < 1e-2
assert abs(row[col["dist_to_obstacle"]] - vec_env.lidar.last_distances.min()) < 1e-3
ok("privileged values are read from the simulator, not approximated")

# On an image-only env the LIDAR is never scanned, so a dist_to_obstacle column
# would be a constant from initialisation. It must be absent, not stale.
img_only = carnav.make(obs_type="image", renderer=renderer, image_size=IMAGE_SIZE)
assert "dist_to_obstacle" not in PrivilegedRecorder(img_only).names
img_only.close()
ok("columns the env cannot supply are dropped, never faked",
   "no dist_to_obstacle without a LIDAR scan")

# ======================================================================
section("4. TEMPORAL ALIGNMENT (deterministic actions)")
# ======================================================================
align_cfg = CollectConfig(capture_rgb=False, capture_privileged_state=True)
align_env = carnav.make(obs_type="vector")
rec = LeWMDatasetEnv(align_env, align_cfg)
obs, info = rec.reset(seed=21)
ACTS = fixed_actions(200, seed=5)
used = []
for a in ACTS:
    obs, r, te, tr, info = rec.step(a)
    used.append(a)
    if te or tr:
        break
arrays = rec.arrays()
priv = arrays["privileged"]
col = {n: i for i, n in enumerate(rec.recorder.names)}
T = len(used)
assert len(arrays["obs_vector"]) == T + 1 and len(arrays["action"]) == T
ok("observation records are T+1 against T action records", f"T = {T}")


def pose_residual(shift):
    """Integrate privileged[t] with action[t+shift]; return the worst error."""
    worst = 0.0
    probe = Car(CarParams())
    for t in range(T - shift):
        probe.reset(priv[t, col["x"]], priv[t, col["y"]], priv[t, col["heading"]],
                    speed=priv[t, col["speed"]])
        probe.steer_angle = float(priv[t, col["steer_angle"]])
        a = arrays["action"][t + shift]
        probe.step(float(a[0]), (float(a[1]) + 1.0) * 0.5, float(a[2]),
                   align_env.cfg.dt)
        worst = max(worst, float(np.hypot(probe.x - priv[t + 1, col["x"]],
                                          probe.y - priv[t + 1, col["y"]])))
    return worst


aligned, shifted = pose_residual(0), pose_residual(1)
assert aligned < 2e-3, f"obs[t] + action[t] does not produce obs[t+1]: {aligned:.2e} m"
assert shifted > 20.0 * aligned, \
    f"the alignment test cannot detect an off-by-one: {shifted:.2e} vs {aligned:.2e}"
ok("obs[t] + action[t] -> obs[t+1], re-integrated through the car model",
   f"residual {aligned:.2e} m")
ok("and the same test with actions shifted one step FAILS, as it must",
   f"{shifted:.2e} m, {shifted / aligned:.0f}x worse")

# Replay: same seed, same stored actions, same observations -- bit for bit.
replay_env = carnav.make(obs_type="vector")
o, _ = replay_env.reset(seed=21)
assert np.array_equal(o, arrays["obs_vector"][0])
for t, a in enumerate(arrays["action"]):
    o, r, te, tr, _ = replay_env.step(a)
    assert np.array_equal(o, arrays["obs_vector"][t + 1]), f"replay diverged at t={t + 1}"
replay_env.close()
align_env.close()
ok("replaying the stored actions from the stored seed is bit-identical")

# ======================================================================
section("5. GOAL BEACON: visual only, off by request, on by default")
# ======================================================================
# One renderer, flag flipped between two captures of the same state: the two
# frames differ in nothing else, which is what makes the comparison evidence.
b_env = carnav.make(obs_type="both", renderer=renderer, image_size=IMAGE_SIZE)
obs, info = b_env.reset(seed=7)
pol = make_policy("competent", b_env, np.random.default_rng(7))
vec_same, diffs, targets0 = True, [], [tuple(t) for t in b_env.targets]
for t in range(250):
    renderer.show_goal_beacon = True
    on = renderer.capture(b_env)
    renderer.show_goal_beacon = False
    off = renderer.capture(b_env)
    diffs.append(int(np.abs(on.astype(np.int16) - off.astype(np.int16)).max()))
    renderer.show_goal_beacon = True
    obs, r, te, tr, info = b_env.step(pol.act(obs, info))
    if te or tr:
        break
assert max(diffs) > 0, "hiding the beacon changed no pixel in 250 steps"
assert min(diffs) == 0, "expected the beacon to be occluded at least sometimes"
ok("show_goal_beacon=False removes the posts and nothing else",
   f"visible in {sum(1 for d in diffs if d > 0)}/{len(diffs)} steps")

# The beacon is a rendering choice. It must not touch the task: same seed, same
# actions, same vector observations, rewards and waypoints either way.
acts = fixed_actions(80, seed=31)
runs = []
for show in (True, False):
    cfg_b = CollectConfig(capture_rgb=True, image_size=IMAGE_SIZE, show_goal_beacon=show)
    r_b = build_renderer(cfg_b)
    e_b = build_env(cfg_b, r_b)
    o, _ = e_b.reset(seed=41)
    vecs, rews = [o["vector"].copy()], []
    for a in acts:
        o, rew, te, tr, _ = e_b.step(a)
        vecs.append(o["vector"].copy())
        rews.append(rew)
        if te or tr:
            break
    runs.append((np.stack(vecs), np.asarray(rews), [tuple(t) for t in e_b.targets]))
    e_b.close()
assert np.array_equal(runs[0][0], runs[1][0]), "beacon changed the vector observation"
assert np.array_equal(runs[0][1], runs[1][1]), "beacon changed the reward"
assert runs[0][2] == runs[1][2], "beacon changed the waypoints"
ok("the beacon is visual only: identical vector obs, rewards and waypoints",
   f"{len(runs[0][1])} steps compared")

# ======================================================================
section("6. DATASET ROUND TRIP, SPLITS AND VIEWS")
# ======================================================================
root = os.path.join(TMP, "ds")
ds_cfg = CollectConfig(capture_rgb=True, image_size=IMAGE_SIZE, n_train=2,
                       n_val=1, n_test=1, max_steps=40, environment_seed=1234)
collect(ds_cfg, root, progress=False)
ds = Dataset(root)
assert ds.manifest["n_episodes"] == 4
assert set(ds.splits) == {"train", "val", "test"}
seen = [set(ds.episode_ids(s)) for s in ("train", "val", "test")]
assert not (seen[0] & seen[1]) and not (seen[0] & seen[2]) and not (seen[1] & seen[2])
seeds = [ds.manifest["episodes"][i]["episode_seed"] for i in range(4)]
assert len(set(seeds)) == 4
ok("splits are disjoint directories with disjoint seeds",
   " / ".join(f"{s}={len(ds.episode_ids(s))}" for s in ("train", "val", "test")))

with ds.episode(ds.episode_ids("train")[0]) as ep:
    T = ep.length
    assert len(ep["timestep"]) == T + 1 and len(ep["action"]) == T
    assert ep["rgb"].shape == (T + 1, IMAGE_SIZE, IMAGE_SIZE, 3)
    full = ds.transitions(ep, "vector_full")
    assert full["obs"].shape == (T, ds.manifest["vector_dim"])
    assert np.array_equal(full["obs"][1:], full["next_obs"][:-1])
    assert "privileged" in full and full["privileged"].shape[0] == T
    assert full["obs"].shape[1] == ds.manifest["vector_dim"], \
        "privileged state must never be concatenated into the model's input"

    keep = ds.restricted_mask()
    lo_nav, hi_nav = ds.manifest["obs_slices"]["nav"]
    lo_dyn, hi_dyn = ds.manifest["obs_slices"]["dynamics"]
    dropped = (hi_nav - lo_nav) + (hi_dyn - lo_dyn)
    assert keep.sum() == ds.manifest["vector_dim"] - dropped
    assert not keep[lo_nav:hi_nav].any() and not keep[lo_dyn:hi_dyn].any()
    restricted = ds.transitions(ep, "vector_restricted")
    assert np.array_equal(restricted["obs"], full["obs"][:, keep])

    pixels = ds.transitions(ep, "rgb")
    assert pixels["obs"].shape == (T, IMAGE_SIZE, IMAGE_SIZE, 3)
    assert np.array_equal(pixels["action"], full["action"])
ok("one set of trajectories, three views, same actions and timing",
   f"full {ds.manifest['vector_dim']}D / restricted {int(keep.sum())}D / "
   f"rgb {IMAGE_SIZE}x{IMAGE_SIZE}")

try:
    write_episode(os.path.join(TMP, "bad.npz"),
                  {"timestep": np.arange(5), "action": np.zeros((5, 3))}, {})
except ValueError:
    ok("writing an episode with a broken length contract raises")
else:
    raise AssertionError("write_episode accepted T+1 != T + 1")

# Determinism of the whole collection, not just of one episode.
root2 = os.path.join(TMP, "ds2")
collect(CollectConfig.from_dict(ds_cfg.to_dict()), root2, progress=False)
ds2 = Dataset(root2)
rgb_stats = []
for eid in ds.episode_ids():
    with ds.episode(eid) as a, ds2.episode(eid) as b:
        for name in a.arrays:
            if name == "rgb":
                # Pixels get a tolerance; everything else must be bit-exact.
                # The Panda3D/EGL rasteriser is not bit-reproducible across
                # runs: a pixel on a polygon edge can go either way depending
                # on submission order, and node creation order is not identical
                # between the first and second collection in a process.
                for t in range(len(a[name])):
                    same, st = rgb_equivalent(a[name][t], b[name][t])
                    rgb_stats.append(st)
                    assert same, f"{eid}:rgb[{t}] differs by {st}"
            else:
                assert np.array_equal(a[name], b[name]), \
                    f"{eid}:{name} is not reproducible"
        assert a.meta["episode_seed"] == b.meta["episode_seed"]
        assert a.meta["policy"] == b.meta["policy"]
n_px = sum(1 for s in rgb_stats if s["pixel_fraction"] > 0)
ok("the same config collected twice gives identical datasets",
   f"every non-pixel array bit-exact; {n_px}/{len(rgb_stats)} frames differ at "
   f"all, worst {max(s['max_channel_diff'] for s in rgb_stats)}/255")

# ======================================================================
section("7. THE VALIDATOR ACTUALLY VALIDATES")
# ======================================================================
quiet = lambda *a, **k: None
good, report = validate_dataset(root, replay=1, log=quiet, strict_coverage=False)
assert good, [c for c in report["checks"] if not c["ok"]]
ok("a freshly collected dataset passes every check",
   f"{len(report['checks'])} checks")

# Corrupt a copy the way a real off-by-one would: roll the actions by one step,
# keeping every array's length and the manifest intact. Nothing structural is
# wrong with the result, which is exactly why the dynamics check has to be the
# thing that catches it.
bent = os.path.join(TMP, "ds_offbyone")
shutil.copytree(root, bent)
bad_ds = Dataset(bent)
for m in bad_ds.manifest["episodes"]:
    p = os.path.join(bent, m["path"])
    with bad_ds.episode(m["episode_id"]) as ep:
        arrays = {n: np.array(ep[n]) for n in ep.arrays}
        meta = dict(ep.meta)
    arrays["action"] = np.roll(arrays["action"], 1, axis=0)
    write_episode(p, arrays, meta)
bad, bad_report = validate_dataset(bent, replay=0, log=quiet, strict_coverage=False)
assert not bad, "the validator passed a dataset with the actions shifted by one"
failed = {c["name"] for c in bad_report["checks"] if not c["ok"]}
assert any("obs[t] + action[t]" in n for n in failed), failed
ok("and rejects the same dataset with its actions shifted by one step",
   f"{len(failed)} check(s) failed, including the alignment check")

b_env.close()
img_env.close()
vec_env.close()
env.close()
shutil.rmtree(TMP, ignore_errors=True)

print()
print("=" * 70)
print(f"ALL {CHECKS[0]} CHECKS PASSED")
print("=" * 70)
