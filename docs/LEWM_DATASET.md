# LeWM dataset: schema, visual setup, and compatibility

Reference for the `lewm/` package: what it records, what the pixel view actually
contains, what varies between episodes, and what it does to the existing
simulator (nothing). Written against *LeWM_Dataset_Creation_Brief.md*; section
numbers in brackets point back at it.

The companion document is the pilot report, which carries the *measured*
numbers for one concrete run: `<dataset>/PILOT_REPORT.md`.

---

## 1. Status: stage 0 only

This is instrumentation plus a small pilot, which is all the brief asks for at
this point [§1, §30]:

| asked for | state |
|---|---|
| configuration switches, collector, schema | done |
| privileged state recorded separately | done |
| validation tool | done (`python -m lewm validate`) |
| small pilot dataset + report | done (12 episodes) |
| compatibility report | section 14 of this file |
| large-scale dataset | **not started**, deliberately |
| pixel world model, latent probes, CEM+MPC planning | **not started**, deliberately |

No model has been trained on this data. The pilot establishes that the
trajectories are *correct* -- aligned, reproducible, complete, separated from
ground truth -- not that they are *sufficient* to learn a world model, which
cannot be established without a training run [§30].

MCTS is explicitly not implemented and should not be [§26]; the planner this
dataset is aimed at is CEM over a learned latent model, which comes after a
model exists.

---

## 2. Quick start

```bash
# a pilot: collect, validate, figures, throughput, report
python -m lewm pilot ~/lewm_runs/pilot_001

# a specific collection
python -m lewm collect ~/lewm_runs/vec_only --n-train 200 --n-val 20 --n-test 20
python -m lewm collect ~/lewm_runs/pixels --rgb --image-size 64 --n-train 200

# the beacon-off control dataset (section 10 below)
python -m lewm collect ~/lewm_runs/pixels_nobeacon --rgb --no-beacon --n-train 200

python -m lewm validate ~/lewm_runs/pixels      # re-check an existing dataset
python -m lewm bench --rgb                      # throughput only
python -m lewm info ~/lewm_runs/pixels          # manifest summary
```

```python
from lewm import CollectConfig, Dataset, collect, validate_dataset

collect(CollectConfig(capture_rgb=True, n_train=8, n_val=2, n_test=2),
        out_dir="runs/pilot")

ds = Dataset("runs/pilot")
for ep in ds.iter_episodes("train"):
    tr = ds.transitions(ep, mode="rgb")       # obs/action/next_obs, aligned
    probe_targets = tr["privileged"]          # never merged into obs
```

---

## 3. Configuration [§3]

One object, `lewm/config.py:CollectConfig`, with explicit keyword defaults and a
JSON round-trip. It follows the convention the repo already has (`EnvConfig`,
`CityConfig`, `CarParams`: plain classes, flat keyword routing through
`carnav.make`, JSON on disk) rather than introducing a second configuration
system [§3]. numpy is this project's only hard dependency and it stays that way.

Simulator settings are *not* duplicated here. They go in `env_kwargs`, forwarded
verbatim to `carnav.make(**env_kwargs)`, so anything the env accepts is
reachable and a typo still raises there instead of being silently ignored.

| flag | default | what it does |
|---|---|---|
| `observation_mode` | `"vector_full"` | which view `Dataset.transitions` returns by default: `vector_full`, `vector_restricted`, `rgb` |
| `restricted_drop` | `("nav", "dynamics")` | observation blocks the restricted view masks out |
| `capture_vector` | `True` | write `obs_vector` |
| `capture_rgb` | `False` | write `rgb` (and switch the env to `obs_type="both"`) |
| `capture_privileged_state` | `True` | write `privileged` |
| `capture_traffic_state` / `capture_signal_state` / `capture_pedestrian_state` | `True` | per-timestep auxiliary ground-truth blocks |
| `n_traffic_state` / `n_pedestrian_state` | 8 / 4 | K for those fixed-width blocks |
| `show_goal_beacon` | `True` | `False` hides the green waypoint posts in the render [§11] |
| `image_size` | 64 | square frame; fixed for a process (see section 5) |
| `camera` | `fov=60, cam_dist=13, cam_height=6.5, look_ahead=9` | the chase rig; recorded in every manifest |
| `environment_seed` | 0 | the one number the whole seed plan derives from |
| `n_train` / `n_val` / `n_test` | 8 / 2 / 2 | episodes per split |
| `max_steps` | `None` | collection-side cap; never changes the env's own limit |
| `policy_mix` | see section 4 | profile shares, must sum to 1 |
| `env_kwargs` | `{}` | forwarded to `carnav.make` |
| `compress` | `False` | `np.savez_compressed` |

Capture flags are deliberately independent of `observation_mode`: the mode says
what a *model* may see, the flags say what is written. Recording RGB while
training on the vector is the whole point of the three-views requirement [§17].

Four combinations are rejected at construction rather than hours later at load
time: `observation_mode="rgb"` without `capture_rgb`, a vector mode without
`capture_vector`, capturing nothing at all, and an unknown camera key.

Episode and environment seeds are both explicit and both recorded [§3]. A
dataset carries the full `CollectConfig` in `manifest.json` and `config.json`,
so any dataset can say exactly how it was made.

---

## 4. Trajectory mixture [§7]

Six profiles, all of them the existing `baselines/scripted.py:GapFollower` with
different gains plus optional action noise -- reusing the controller rather than
writing new ones, as the brief asks. Shares are dataset-level and apportioned by
largest remainder over `n_episodes`, then dealt to the splits, so the
composition the manifest claims is the composition on disk even at pilot scale.

| profile | share | what it is | coverage it buys |
|---|---|---|---|
| `competent` | 25% | `GapFollower` defaults | the nominal manifold |
| `noisy_competent` | 15% | defaults + sigma 0.15 action noise | the tube around it |
| `aggressive` | 25% | cruise 16 m/s, late braking, loose steering | high speed, sharp turns, near misses |
| `intermediate` | 10% | mis-tuned gains, sigma 0.25 noise | weaving, overshoot, late correction |
| `conservative` | 20% | cruise 6 m/s, early braking, big gaps | low speed, long follows, creeping |
| `recovery` | 5% | competent + injected off-nominal bursts | hard brake / oversteer **and the recovery from it** |

Why not random actions, the usual world-model default: the action space is
`[signed throttle, brake, steer]` with brake rescaled from `[-1, 1]` to
`[0, 1]`, so a uniform random action has `E[throttle] = 0` and `E[brake] = 0.5`
-- the car idles. `tests/test_env.py` scores random at 0.00/3 waypoints and
-104 return against the scripted driver's 1.55 and +168. A dataset of
mostly-stationary trajectories teaches a world model that the world does not
move.

Why not competent driving only: a world model is asked "what happens if I do
*this* from *here*", and the states a planner proposes mid-search are not the
states a good driver visits. Hence the deliberate departures, and hence
`recovery`, which injects a sustained wrong input (hard brake, oversteer,
throttle spike) for 8-22 steps every 60-140 steps and then hands back to the
competent driver. Both halves are recorded and the burst steps are labelled
`injected`, so off-nominal data is identifiable rather than just present.

Every profile consumes **only the observation vector**. The demonstrator never
reads privileged state, so it cannot leak into the data through the actions.

---

## 5. Schema

### Layout

```
<root>/
    manifest.json          config, schema version, split index, obs_slices, per-episode meta
    config.json            the CollectConfig that produced it
    episodes/
        train/ep_000000.npz
        val/ep_000009.npz
        test/ep_000011.npz
    samples/               figures (pilot only)
    PILOT_REPORT.md        (pilot only)
    validation.json        (pilot only)
```

Splits are **directories**, not a column: a training frame cannot appear in the
test set without moving a file [§16]. The manifest repeats the mapping and the
validator checks the two agree in both directions.

One `.npz` per episode, each self-describing -- it carries its own `meta` JSON
next to the arrays, so a single episode is readable and checkable without the
manifest. Whole episodes rather than flattened transitions because LeWM trains
on temporal context [§14, §17]; a shuffled transition store cannot hand back a
contiguous run, which is also why `replay/buffer.py` already has an
`EpisodeBuffer` beside its `ReplayBuffer`.

### Temporal alignment [§13]

The one convention everything depends on:

```
observation-side records have length T+1
action-side records have length T
transition t is  (obs[t], action[t]) -> obs[t+1]
```

So `obs_vector[t]`, `rgb[t]` and `privileged[t]` are all the state the policy
*saw* when it chose `action[t]`, and index `t+1` is the consequence. There is no
`next_obs` array on disk: storing one would double the RGB and create a second
place for exactly the off-by-one error the brief is worried about.
`transitions()` builds the triples by slicing.

Two further guards, both of them structural rather than conventional:

* When RGB is captured the env runs with `obs_type="both"`, so the frame and the
  vector come out of a single `_observe()` call inside the env. Capturing the
  frame with a separate `env.render()` after `step()` would also work today, but
  it would put the alignment guarantee in `lewm/` instead of in the env.
* `write_episode` checks the length contract at the one place episodes are
  created and raises rather than writing a file that violates it.

### Arrays

Observation-side, length T+1:

| array | dtype | shape | present when |
|---|---|---|---|
| `timestep` | int32 | (T+1,) | always |
| `obs_vector` | float32 | (T+1, D) | `capture_vector` |
| `rgb` | uint8 | (T+1, H, W, 3) | `capture_rgb` |
| `privileged` | float32 | (T+1, P) | `capture_privileged_state` |
| `traffic_state` | float32 | (T+1, 8, 6) | traffic on + `capture_traffic_state` |
| `pedestrian_state` | float32 | (T+1, 4, 5) | pedestrians on + `capture_pedestrian_state` |
| `signal_state` | float32 | (T+1, S, 3) | lights on + `capture_signal_state` |

Action-side, length T:

| array | dtype | shape | notes |
|---|---|---|---|
| `action` | float32 | (T, 3) | exactly what `env.step` was given, already clipped |
| `reward` | float32 | (T,) | the env's own reward, unmodified |
| `terminated` | bool | (T,) | true only on the final step of a terminated episode |
| `truncated` | bool | (T,) | likewise for a timeout |
| `crashed` | bool | (T,) | `info["crashed"]` |
| `reached` | int8 | (T,) | `info["targets_reached_this_step"]` |
| `injected` | bool | (T,) | the step was an off-nominal burst, not the controller's choice |

`meta` (JSON, per episode): `episode_id`, `episode_index`, `split`, `policy`,
`environment_seed`, `episode_seed`, `policy_seed`, `length`, `reason`,
`episode_reward`, `targets_reached`, `red_light_violations`, `crash_with`,
`targets` (the full waypoint list in world metres), `signal_xy`, `obs_slices`,
`privileged_names`, `block_columns`, `camera`, `show_goal_beacon`, `dt`,
`action_repeat`, `max_episode_steps`, `n_targets`, `vector_dim`, `image_size`,
`collected_at`, `wall_s` and `steps_per_s`.

`manifest.json` adds the dataset-level view: the full `CollectConfig`, the split
index, `obs_slices`, `vector_dim`, `privileged_names`, `block_columns`,
`n_episodes`, `n_transitions`, measured `throughput`, and an `environment` block
with the `git_sha`, platform and library versions the data was collected on.

`reason` ∈ `success`, `crash`, `timeout`, `stuck`, `cut`. `cut` is the
collection-side `max_steps` cap, and it sets neither `terminated` nor
`truncated` on its last step, so a consumer bootstrapping values can tell a cut
from a real timeout.

### The vector observation

73 channels with the env's defaults. Never hard-code those offsets: the layout
is `CarNavEnv.obs_slices`, it moves with `n_beams` / `n_lookahead` /
`n_traffic_obs` / the subsystem switches, and it is recorded per dataset.

| block | width (default) | contents |
|---|---|---|
| `lidar` | 32 | normalised beam distances, 360° |
| `dynamics` | 5 | speed, yaw rate, steer angle, accel, slip |
| `nav` | 9 | 3 lookahead waypoints x (distance, sin/cos bearing) |
| `traffic_light` | 7 | nearest signal: distance, phase one-hot, timing |
| `traffic` | 20 | 4 nearest vehicles x 5 |

---

## 6. Privileged state [§5, §6, §24]

Recorded for **evaluation only**: the half of the dataset the pixel model must
never see, so a latent learned from RGB + action can afterwards be probed for
position, heading, velocity, goal distance and obstacle distance using numbers
the model was never given.

Two rules shape it.

**Only quantities the simulator actually has** [§6]. Every column is read
straight off `env.car`, `env.targets`, `env.traffic`, `env.street`,
`env.traffic_lights` or `env.lidar`. Nothing is estimated, smoothed or
back-differenced. A quantity whose subsystem is off is *absent*, not present and
filled with a placeholder -- a probe trained against a constant column reports a
meaningless R² instead of failing loudly. There are no pedestrian columns in a
dataset collected without pedestrians, and no `dist_to_obstacle` in an
image-only dataset (nothing requested a LIDAR scan that step, so the stored
value would be stale).

**The column set is fixed at construction and asserted on every row.** A row of
the wrong width is a corrupted dataset, so it raises before being written.

Default columns (26 with the env's defaults: pedestrians and speed signs off):

| group | columns |
|---|---|
| ego pose and motion (world frame) | `x`, `y`, `heading`, `speed`, `vx`, `vy`, `accel`, `yaw_rate`, `steer_angle`, `slip` |
| task | `target_idx`, `dist_to_goal`, `goal_x`, `goal_y`, `goal_bearing` |
| sensing (vector/both only) | `dist_to_obstacle` |
| traffic (if on) | `dist_to_vehicle`, `bearing_to_vehicle`, `vehicle_speed`, `dist_to_moving_vehicle`, `bearing_to_moving_vehicle`, `moving_vehicle_speed` |
| signals (if on) | `dist_to_signal`, `signal_state`, `signal_steps_remaining`, `red_light_violations` |
| pedestrians (if on) | `dist_to_pedestrian`, `bearing_to_pedestrian`, `pedestrian_speed`, `pedestrians_on_road` |
| signs (if on) | `speed_limit` |

`vx`/`vy` use `heading + slip`, matching how the bicycle model actually
integrates position, rather than projecting on heading alone.

Sentinel: a distance or bearing for an object that does not exist *at this
timestep* is `-1.0`, outside the range of every real value in those columns.
Filter on `>= 0` before probing.

Separation is structural. The privileged arrays live in their own keys, the
manifest names them separately, `Dataset.transitions` returns them under
`privileged` / `next_privileged` and **never** concatenates them into `obs`.
Handing ground truth to a model takes a deliberate line of code.

The `info` dict already exposes a handful of these and INTEGRATION.md already
labels them "privileged ground truth, for logging and diagnostics only"; this is
that same idea widened to what a probing experiment needs.

---

## 7. The three views [§17]

All three come from the *same* recorded trajectories -- same seeds, same
actions, same cities -- so a comparison between them is a comparison of
observation content and nothing else.

| view | obs | dim (defaults) | the question it answers |
|---|---|---|---|
| `vector_full` | the env's vector observation | 73 | can a latent model learn the dynamics at all, given clean state? |
| `vector_restricted` | with `nav` and `dynamics` masked out | 59 | how much of the task was the explicit navigation/dynamics state doing? |
| `rgb` | the 64x64 frame | 64x64x3 | can it be learned from pixels? |

The restricted mask is computed from the `obs_slices` recorded with the dataset,
not from hard-coded offsets -- the mistake `baselines/scripted.py` documents
having already made once in this repo. `Dataset.restricted_mask(drop=...)` takes
any set of block names if you want a different ablation.

A model trained on one view can be evaluated against the privileged state of the
same episodes, which is the point of recording all three from one collection.

---

## 8. Splits and seeds [§16]

```
SeedSequence(environment_seed)
    |- train block -> per-episode child -> (episode_seed, policy_seed)
    |- val block   -> ...
    |- test block  -> ...
    |- mixture block -> the policy schedule
```

* Splits are assigned at the **episode / environment-seed level**, never at the
  frame level [§16]. Frames from one episode are all in one split, by file
  location.
* Train, val and test draw from *different spawned blocks*, so a seed cannot
  collide across splits by construction -- and the validator checks it anyway.
* The mixture draws from its own block, so changing the policy shares does not
  move which cities the episodes are collected in.
* Each episode's seeds are a pure function of `(environment_seed, split, index)`,
  which makes a large collection shardable: N processes can each take a stride
  of episode indices with no coordination and no seed reuse.

Each episode is a freshly generated city (`randomize_map=True` by default), so
train and test differ in layout, not just in starting pose.

---

## 9. The visual setup [§12]

What the pixel view actually is, since the brief asks for this to be written
down rather than inferred later from the images.

### The rig

| property | value | note |
|---|---|---|
| resolution | 64x64 RGB uint8 | `image_size`; the renderer's own default |
| camera | **chase**, rigidly attached to the car | not a fixed world camera |
| position | 13.0 m behind the car, 6.5 m up | `cam_dist`, `cam_height` |
| aim | 9.0 m ahead of the car, at z = 1.2 m | `look_ahead` |
| field of view | 60° | `fov` |
| near / far plane | 0.3 m / 400 m | fixed in `PandaRenderer` |
| frame rate | 20 Hz (`dt=0.05`, `action_repeat=1`) | one frame per env step |
| frames per episode | up to 1001 | `max_episode_steps=1000`, plus the reset observation |
| lighting | one ambient + two directional (sun + fill), fixed | no time of day, no shadows |
| sky / fog | constant colour, exponential distance fog | no weather variation |
| camera smoothing | none | pose is a pure function of `(x, y, heading)` |

The camera is a deterministic function of the car's pose with no filtering or
lag, so a frame is a function of the world state alone -- which is what makes
the validator's "collect the same config twice and compare" check meaningful.

A single `PandaRenderer` per process is the renderer's standing contract
(INTEGRATION.md), and the offscreen buffer size is fixed when the process's
`ShowBase` is first built, so `image_size` cannot change within a process.
`lewm.collector.build_renderer` keeps one process-wide instance and reconfigures
its camera and beacon flag in place; constructing a second one does not replace
the first, it *adds* to the same scene graph (double lighting, a ghost car), and
that is a real bug this code hit and fixed, not a hypothetical.

### What is in the frame

Roads, buildings, raised kerbs, parked and moving vehicles, pedestrians and
zebra stripes when enabled, traffic-light heads, and the ego car itself. The ego
occupies a fixed region of the image, since the camera is rigidly attached to
it. There are **no lane markings and no centre line** -- the correct side of the
road is not visually marked, so a pixel model has no cue for it beyond where
other vehicles are.

One deliberate rendering choice matters for world-model work and is easy to miss:
the road surface is emitted as one quad per tile with per-tile shading rather
than as a single plane (`render/panda_renderer.py:662`), *because* a featureless
plane gives almost no optical flow. The ground texture is what makes speed
recoverable from a pair of frames at all.

### What is *not* recoverable from a single frame

Worth knowing before probing a single-frame latent for it:

* **Speed, yaw rate, acceleration.** Nothing in the image encodes them -- no
  speedometer, no motion blur. They are inferable only from a stack of frames.
  A single-frame probe for velocity *should* fail; that is not a bug in the
  dataset.
* **Steering angle.** The car model is placed by position and heading only; its
  front wheels do not visibly turn.
* **Anything behind the car.** The camera faces forward with a 60° FOV. The
  LIDAR block of the vector observation covers 360°, so the pixel view is
  strictly less informative here -- relevant when comparing views, since a
  vehicle approaching from behind is observable in one view and not the other.
* **Traffic-light phase at distance.** The lamp heads are a handful of pixels at
  64x64; the vector observation gets the phase as a clean one-hot. Expect the
  pixel model to be much worse at signals specifically.

### Position leakage

The camera is ego-centric, so absolute world position is not *directly* encoded.
It is still **inferable**, and by design: `render/panda_renderer.py:_hash01`
derives every building's height and shade from its tile coordinates rather than
from an RNG, so "a given tile looks the same every time it appears". The skyline
is therefore a fingerprint of absolute map position, consistent across episodes
and across map layouts.

Consequence for probing: a position probe on a pixel latent can succeed by
recognising buildings rather than by integrating motion, and from a *single
frame*, which is the signature to look for. If that distinction matters for an
experiment, the cheap control is to compare the probe's accuracy on held-out
episodes against its accuracy on positions that never appeared in training.
Nothing here needs changing to run that control -- but the leak should not be
discovered after the fact, hence this paragraph.

### Other shortcuts available to a pixel model

* The goal beacon, quantified in section 10.
* Road geometry is a perfectly regular grid with 12 m corridors, so "drive
  straight until a gap appears" is a strong prior that generalises across maps.
* Per-tile ground shading gives strong optical flow on a regular grid, so
  speed-from-flow is an easier signal here than on real road surfaces.
* Buildings are untextured flat-shaded boxes: there is very little visual
  nuisance variation, which makes pixel learning easier here than on real
  imagery and the results correspondingly less transferable.

---

## 10. Goal representation and the beacon [§11]

The goal reaches the agent through two channels.

1. **The `nav` block** of the vector observation: distance and bearing to the
   next three waypoints. Explicit, and dropped by the `vector_restricted` view.
2. **The visual beacon**: tall green posts rendered at the next waypoints. This
   is the one that matters for a pixel model, and the brief asks for it to be
   switchable rather than assumed.

`show_goal_beacon=False` (CLI `--no-beacon`) hides the posts. It is an
**experimental configuration, not a removal**: the flag lives on the injected
`PandaRenderer`, defaults to `True`, and no code under `env/` changed. Every
existing screenshot, checkpoint and viewer still renders exactly as before
[§11]. `tests/test_lewm.py` checks that a beacon-on and a beacon-off run produce
identical vector observations, rewards and waypoints -- the flag touches pixels
and nothing else.

Measured, by capturing every step twice with the flag flipped between the two
captures of the *same* simulator state, over 800 steps of competent driving:

* the beacon changes at least one pixel in **28.0%** of steps
* when visible it covers **0.80%** of the frame on average, at most **5.37%**

The waypoints are 40-110 m away when sampled and the city's buildings are 6-24 m
tall, so the posts are occluded for most of a run and appear once the car is on
the right street. So the beacon is not a free answer to the navigation problem
-- but it is a strong, highly salient local cue (a full-height green column)
exactly when the goal is nearly reached, which is the part of the task a pixel
world model would most plausibly shortcut.

The recommendation is to collect both and compare, not to pick one: the two
datasets differ in nothing but the posts, and the difference in a model's
goal-reaching behaviour between them *is* the measurement of how much the
beacon was doing. See `<dataset>/samples/beacon_ab.png` for the A/B figure and
`beacon_ab.json` for the numbers.

---

## 11. Environment variation [§10]

### What varies per episode (driven by `episode_seed`)

| varies | detail |
|---|---|
| city layout | road positions, block sizes, plazas, walled-off segments -- regenerated every reset (`randomize_map=True`) |
| which crossings are signalised | up to 6, min 45 m apart |
| signal phases and timers | staggered per light at reset |
| ego spawn pose | sampled free road pose, any heading |
| waypoints | 3, chained, each 40-110 m from the last |
| moving traffic | 8 vehicles: routes, speeds (9 m/s jittered), spawn positions |
| parked vehicles | 18, at kerbside spots |
| pedestrians | when enabled: 6 crossings, 1-3 each |
| LIDAR / radar noise draws | when the noise parameters are non-zero (default 0) |

### What is fixed across every episode

| fixed | value |
|---|---|
| city extent | 64 x 64 tiles x 4.0 m = 256 x 256 m |
| road width | 3 tiles = 12 m |
| generator distribution | `CityConfig` defaults -- the seed varies, the generator does not |
| car parameters | `CarParams` defaults (max speed 22 m/s, min turn radius, mass...) |
| physics timestep | `dt=0.05`, `action_repeat=1` |
| episode limit | 1000 steps |
| reward weights | `EnvConfig` defaults |
| building appearance | a deterministic hash of tile coordinates (see section 9) |
| lighting, sky, fog | constant; no weather or time of day |
| camera rig | one configuration per dataset, recorded in the manifest and checked |

So the variation this dataset exercises is **layout, traffic and task
variation**, not appearance or dynamics variation. A model trained on it should
be expected to generalise across cities, and should *not* be expected to
generalise across lighting, weather, car parameters or camera placement, none of
which vary. Changing any of them is an `env_kwargs` / `camera` away, and the
validator enforces one camera configuration and one beacon setting per dataset
so two incompatible visual regimes cannot be mixed into one training set by
accident.

---

## 12. Validation [§19]

`python -m lewm validate <root>` -- 37 checks in six groups on the pilot
configuration. It re-runs the simulator, so it is a real re-derivation, not a
file-format check.

| group | what it establishes |
|---|---|
| structural | manifest ↔ disk agree both ways; ids and seeds unique; splits disjoint; no seed shared across splits; obs-side == act-side + 1; timesteps contiguous from 0; lengths within the limit; done flags only on the final step and consistent with `reason`; no NaN/Inf; every promised array present |
| temporal | the two checks below |
| visual | one consistent frame shape and dtype; one camera configuration; one beacon setting; no flat or corrupt frames; frames change while the car moves |
| action | inside the action space; no degenerate channel; both steering directions; both ends of the brake range; full forward throttle |
| dynamics | speed/accel/steer/yaw-rate distributions non-degenerate; per-step displacement physically plausible against `max_speed * dt` |
| coverage | share of steps in each behavioural regime against a floor, plus crash/near-miss/obstacle-proximity counts |

The two load-bearing checks, both aimed at [§13]:

1. **Replay.** Rebuild the env from the manifest, `reset(seed=episode_seed)`,
   feed back the stored actions. Vector observations, rewards and privileged
   state must come back **bit-identical**.
2. **Forward dynamics, with a control.** Take the recorded pose at `t`,
   integrate one step with the recorded `action[t]` through the car's own
   kinematic model, compare against the recorded pose at `t+1`. Then do it again
   with the actions shifted by one step. Aligned residual must be at float32
   noise (measured ~1e-5 m) and the shifted one must be orders of magnitude
   worse (measured ~2500x). A check that only passes when correct is weak; this
   one also demonstrably fails when the data is wrong, which is why
   `tests/test_lewm.py` feeds the validator a deliberately `np.roll`-ed copy of
   a good dataset and requires rejection.

RGB is the one exception to bit-exactness. Repeated renders of the same state
are occasionally not pixel-identical: a polygon-edge rasteriser tie-break can
flip a single pixel by a few levels, intermittently, depending on submission
order. So frames are compared with a tolerance -- `rgb_equivalent`: at most
8/255 per channel, on at most 0.1% of pixels -- deliberately tight on *extent*
rather than magnitude, so the duplicated-lighting failure mode that motivated
the process-wide renderer would still be caught. Everything else (vector,
privileged, action, reward) remains bit-exact with no tolerance.

Coverage floors are claims about the *mixture*, so `--no-strict-coverage`
downgrades them to notes for an intentionally narrow collection (a single
profile, a short run) while keeping every correctness check strict.

---

## 13. Throughput and storage [§21]

Measured per capture configuration, cumulatively -- each row adds one cost to
the row above, so differences are attributable. `env.reset` is excluded from the
step rate and reported separately. From the pilot on this box (single process,
headless EGL):

| configuration | steps/s | reset (ms) | KiB/step |
|---|---|---|---|
| `env.step` only (constant action) | 671 | 33 | - |
| + scripted policy | 550 | 33 | - |
| + vector recording | 542 | 33 | 0.3 |
| + RGB capture | 376 | 59 | 12.4 |
| + privileged state | 370 | 59 | 12.7 |

Three things this says:

* **The scripted driver, not the simulator, is the cost of a vector-only
  collection** (671 → 550). Recording on top of it is nearly free (550 → 542).
* **RGB costs about 1.5x in wall-clock and 40x in bytes.** At ~370 steps/s, one
  million RGB transitions is ~0.8 h of compute and ~13 GB uncompressed in a
  single process. That is the number that sizes the real collection, and the
  reason the seed plan was built to shard.
* **Reset nearly doubles with RGB** (33 → 59 ms) because the scene is rebuilt
  per episode. Short episodes pay this disproportionately.

Run-to-run spread on these is roughly ±5%, and more if anything else is using
the box, so treat the ordering as the result and the absolute numbers as
approximate.

Re-measure with `python -m lewm bench [--rgb]` before planning a large run;
these are one machine's numbers.

---

## 14. Compatibility report [§28, §29.7]

> The hard constraint [§2, §27]: do not break, rewrite, or change the existing
> environment's current functionality. What follows is the evidence, not a
> promise.

This package was first written inside `Car-Navigation-Env` and moved out to
its own repository (`lewm-car-nav`, this one) once stage 0 was working, for the
same reason `dreamer-car-nav` is separate from the 2D env: the simulator is a
shared, stable artifact with its own test suite, and this is one of several
possible consumers. The compatibility question therefore spans two
repositories now; both halves are reported below.

### What changed in `Car-Navigation-Env` (the simulator)

| file | change | effect when unused |
|---|---|---|
| `render/panda_renderer.py` | added `show_goal_beacon=True` constructor keyword and the branch that honours it in `_place_actors` | none: the default is the old behaviour, posts drawn |

That is the only edit to pre-existing code, and it is still the only one --
moving the dataset package out did not touch it again. Everything else that
used to live alongside it (`lewm/`, `tests/test_lewm.py`, this document) is
gone from that repository; `README.md` there now points at this one.

Nothing under `env/`, `render/`, `baselines/`, `agents/` or `replay/` imports
`lewm`, and after the move nothing *can*: the package is not even checked out
in the same tree. The dependency arrow points one way by construction, not
just by convention -- the same composition discipline `replay/wrapper.py` and
`wrappers.py` already followed inside that repo.

### What this repository (`lewm-car-nav`) adds

Entirely new files, dependent on the simulator but never imported by it:
`lewm/` (`paths`, config, collector, dataset, policies, state, validate,
pilot, CLI), `tests/test_lewm.py`, and `docs/`. `lewm/paths.py` is the only
glue the split needed -- it finds the sibling checkout at import time
(`CARNAV_ROOT`, then an already-importable `carnav`, then
`../Car-Navigation-Env`) and is the one place that would need to change if the
simulator ever moved or were installed as a package instead of found as a
sibling.

### Checklist [§28]

| requirement | status |
|---|---|
| existing training/eval entry points unchanged | yes -- no file under `env/`, `baselines/`, `agents/`, `replay/` modified |
| default observation space unchanged | yes -- 73-D with defaults, verified in `tests/test_lewm.py` |
| default reward function unchanged | yes -- no reward code touched; `reward` is recorded verbatim, never modified |
| default rendering unchanged | yes -- `show_goal_beacon` defaults to `True`, which is the prior behaviour |
| recording is side-effect free | yes -- **bit-identical rollout with and without recording** over 120 fixed actions from one seed |
| existing configs still load | yes -- `carnav.make` keyword routing and its `TypeError` on unknown keys both re-checked |
| no new hard dependency | yes -- numpy only; `panda3d` stays optional and is needed only for `capture_rgb` |
| new behaviour disabled by default | yes -- `capture_rgb=False`, and the whole package is inert unless imported |
| new behaviour tested in isolation | yes -- `tests/test_lewm.py`, 27 checks in 7 sections |
| simulator checkout independently cloneable/testable | yes -- this repo's import failing (no sibling checkout, no `CARNAV_ROOT`) does not touch the simulator at all |

Datasets record *both* repositories' commits in `manifest["environment"]`
(`git_sha` for this repo, `carnav_git_sha` for the simulator) -- reproducing a
run needs the state of each.

### Existing test suites

Run on this box with the `t3d` conda env, after the changes:

| suite | result |
|---|---|
| `tests/test_core.py` | PASS |
| `tests/test_env.py` | PASS |
| `tests/test_render.py` | PASS |
| `tests/test_replay.py` | PASS |
| `tests/test_street.py` | PASS |
| `tests/test_perception.py` | PASS |
| `tests/test_integration.py` | **FAIL** -- pre-existing, see below |
| `tests/test_lewm.py` (new) | PASS, 27 checks |

### Pre-existing failure, unrelated to this work

`tests/test_integration.py` fails one check:

```
[FAIL] every step's reward matches its declared components -- max residual 7.62e-03 over 25688 steps
```

This failure is present on the commit this work started from, before any file
was touched. It is a reward-accounting residual inside the existing env and has
nothing to do with dataset collection; it is recorded here as the baseline state
rather than fixed, because fixing it would mean changing existing reward
behaviour, which [§2] forbids. Anyone picking that up should treat it as its own
task.

### One other discrepancy worth recording

`README.md` quotes roughly 2,440 steps/s for the vector env. The exact
configuration `tests/test_env.py` benchmarks measures **~584 steps/s** on this
machine (and ~613 at the default 64x64 city). The ratio is consistent across
every configuration measured, so this is a slower machine, not a regression
introduced here -- but a reader comparing section 13's numbers against the
README should know the baseline differs by ~4x.

---

## 15. What this does not establish

* **Nothing has been trained.** Dataset correctness is checked; dataset
  sufficiency for learning a world model is not, and cannot be without a
  training run [§30].
* **12 episodes cannot establish the tail.** Rare events (red-light violations,
  pedestrian conflicts, reverse manoeuvres) appear a handful of times at most,
  so their frequency in the pilot says nothing about their frequency at scale.
* **The map distribution is fixed.** The pilot varies the seed, not the
  generator; see section 11.
* **No appearance variation at all** -- one lighting setup, one camera rig, one
  visual style.
* **Throughput is one machine's.** See section 13.

---

## 16. Files

| file | what it is |
|---|---|
| `lewm/paths.py` | finds the sibling `Car-Navigation-Env` checkout; the only glue the repo split needed |
| `lewm/config.py` | `CollectConfig`, camera defaults, the policy mixture |
| `lewm/collector.py` | `LeWMDatasetEnv` recording wrapper, `collect`, the seed plan |
| `lewm/dataset.py` | on-disk schema, writer, reader, splits, the three views |
| `lewm/policies.py` | the six profiles and the off-nominal injector |
| `lewm/state.py` | `PrivilegedRecorder` -- ground truth, evaluation only |
| `lewm/validate.py` | the validation tool |
| `lewm/pilot.py` | pilot run, benchmark, figures, report |
| `lewm/__main__.py` | `python -m lewm collect|pilot|validate|bench|beacon|info` |
| `tests/test_lewm.py` | compatibility, configuration, privileged state, alignment, beacon, round trip, validator |
| `docs/LEWM_DATASET.md` | this file |
