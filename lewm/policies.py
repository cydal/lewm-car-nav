"""
The trajectory mixture: six driving styles built from the one scripted driver.

Why not a random policy, which is what a world-model dataset usually starts
with: the action space here is `[throttle_signed, brake, steer]` with brake
rescaled from `[-1, 1]` to `[0, 1]`, so a uniform random action has
`E[throttle] = 0` and `E[brake] = 0.5`. The car spends most of a random episode
standing still -- `tests/test_env.py` scores the random policy at 0.00 of 3
waypoints and a mean return of -104 against the scripted driver's 1.55 and +168.
A dataset of mostly-stationary trajectories teaches a world model that the
world does not move.

Why not pure competent driving either: a world model is asked "what happens if
I do *this* from *here*", and a dataset containing only the states a good
driver visits cannot answer that for the states a planner will actually propose
mid-search. So: competent driving as the backbone, and deliberate departures
from it for coverage.

Every profile is the same `GapFollower` with different gains, plus optional
action noise. Re-using the existing controller rather than writing six is the
brief's section 7 instruction and it also means the whole mixture inherits one
property that matters: the driver consumes *only the observation vector*, never
privileged state, so the demonstrator never leaks information the dataset is
supposed to keep separate.

The profiles, and what each is for:

| profile           | what it is                              | coverage it buys                     |
|-------------------|-----------------------------------------|--------------------------------------|
| `competent`       | `GapFollower` defaults                  | the nominal manifold                 |
| `noisy_competent` | defaults + sigma 0.15 action noise      | the tube around it                   |
| `aggressive`      | cruise 16 m/s, late braking, loose steer| high speed, sharp turns, near misses |
| `intermediate`    | mis-tuned gains, sigma 0.25 noise       | sloppy lane-keeping, overshoot       |
| `conservative`    | cruise 6 m/s, early braking, big gaps   | low speed, long follows, creeping    |
| `recovery`        | competent + injected off-nominal bursts | hard brake / oversteer *and the
                                                       recovery back from it* |

`recovery` is the one that is not just a gain change. A world model trained
only on states a controller chooses never sees the state it would be in two
steps after a planner's bad idea, which is exactly where latent rollouts get
evaluated. So the injector takes the wheel for a short burst (hard brake,
excessive steering, or a throttle spike), then hands straight back to the
competent driver, which then has to recover -- and both halves of that are in
the data, labelled, via `info_extra["injected"]`.

Determinism: every profile's randomness comes from the generator handed to
`make_policy`, which the collector derives from the episode seed. Same seed,
same trajectory -- that is what makes the pilot's replay check meaningful.
"""

import numpy as np

from baselines.scripted import GapFollower


# kwargs overrides for GapFollower, plus the action-noise sigma applied on top.
# `None` for `inject` means "no off-nominal bursts".
PROFILES = {
    "competent": {
        "kwargs": {},
        "noise": 0.0,
    },
    "noisy_competent": {
        "kwargs": {},
        "noise": 0.15,
    },
    "aggressive": {
        # Late, hard braking and a quick wheel: the car arrives at corners too
        # fast and has to use the full steering range to make them.
        "kwargs": {"cruise_speed": 16.0, "brake_decel": 8.0, "stop_margin": 1.5,
                   "w_clear": 0.40, "steer_smooth": 0.25, "lookahead_time": 1.0,
                   "follow_gap": 2.0, "conflict_pad": 0.8, "tl_stop_margin": 0.5},
        "noise": 0.10,
    },
    "intermediate": {
        # Plausibly mis-tuned rather than broken: a short pure-pursuit lookahead
        # and a heavy steering filter give the weaving, late-correcting line of a
        # learner, which is the off-nominal-but-recoverable band we want.
        "kwargs": {"cruise_speed": 12.0, "lookahead_time": 0.8, "lookahead_min": 4.0,
                   "k_center": 0.2, "steer_smooth": 0.70, "w_goal": 1.4,
                   "w_clear": 0.35},
        "noise": 0.25,
    },
    "conservative": {
        "kwargs": {"cruise_speed": 6.0, "brake_decel": 4.0, "stop_margin": 6.0,
                   "follow_gap": 7.0, "conflict_horizon": 4.0, "steer_smooth": 0.6,
                   "tl_stop_margin": 4.0},
        "noise": 0.0,
    },
    "recovery": {
        "kwargs": {},
        "noise": 0.05,
        "inject": {"period_steps": (60, 140), "burst_steps": (8, 22)},
    },
}


class ScriptedProfile:
    """A `GapFollower` with per-profile gains, action noise and optional bursts.

    Satisfies the `agents/base.py` contract (`reset`, `act`, `diagnostics`) by
    duck typing, like `GapFollower` itself. `act` returns the action *already
    clipped to the action space*, so what the collector records is exactly what
    the env is given -- clipping inside `CarNavEnv.step` would otherwise make
    the stored action differ from the applied one on every noisy step.
    """

    def __init__(self, name, env, rng, kwargs=None, noise=0.0, inject=None):
        self.name = name
        self.rng = rng
        self.noise = float(noise)
        self.driver = GapFollower.for_env(env, **(kwargs or {}))
        self.inject = dict(inject) if inject else None
        self._burst_left = 0
        self._burst_kind = None
        self._burst_action = None
        self._next_burst = self._draw_next_burst()
        self._step = 0
        self.last_injected = False

    # ------------------------------------------------------------------
    def _draw_next_burst(self):
        if not self.inject:
            return None
        lo, hi = self.inject["period_steps"]
        return int(self.rng.integers(lo, hi + 1))

    def _start_burst(self):
        lo, hi = self.inject["burst_steps"]
        self._burst_left = int(self.rng.integers(lo, hi + 1))
        self._burst_kind = str(self.rng.choice(["hard_brake", "oversteer", "throttle_spike"]))
        if self._burst_kind == "hard_brake":
            a = [0.0, 1.0, float(self.rng.uniform(-0.2, 0.2))]
        elif self._burst_kind == "oversteer":
            a = [float(self.rng.uniform(0.2, 0.6)), -1.0,
                 float(self.rng.choice([-1.0, 1.0]) * self.rng.uniform(0.7, 1.0))]
        else:
            a = [1.0, -1.0, float(self.rng.uniform(-0.3, 0.3))]
        # Held for the whole burst: a sustained wrong input is what actually
        # takes the car off the nominal manifold. Re-sampling every step would
        # average back out to roughly nothing.
        self._burst_action = np.asarray(a, dtype=np.float32)

    # ------------------------------------------------------------------
    def reset(self):
        self.driver.reset()
        self._step = 0
        self._burst_left = 0
        self._burst_kind = None
        self._next_burst = self._draw_next_burst()
        self.last_injected = False

    def act(self, obs, info=None):
        vec = obs["vector"] if isinstance(obs, dict) else obs
        base = np.asarray(self.driver.act(vec, info), dtype=np.float32)

        if self.inject is not None:
            if self._burst_left > 0:
                self._burst_left -= 1
                self.last_injected = True
                self._step += 1
                # Still asks the driver for an action above, so its steering
                # filter keeps tracking the world and hands back something
                # sensible the moment the burst ends.
                return np.clip(self._burst_action, -1.0, 1.0)
            if self._next_burst is not None and self._step >= self._next_burst:
                self._start_burst()
                self._next_burst = self._step + self._draw_next_burst() + self._burst_left
                self._burst_left -= 1
                self.last_injected = True
                self._step += 1
                return np.clip(self._burst_action, -1.0, 1.0)

        action = base
        if self.noise > 0.0:
            action = base + self.rng.normal(0.0, self.noise, size=3).astype(np.float32)
        self.last_injected = False
        self._step += 1
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def diagnostics(self):
        d = dict(self.driver.diagnostics())
        d["profile"] = self.name
        d["injected"] = self.last_injected
        return d


def make_policy(name, env, rng):
    """Build the named profile for `env`, drawing its randomness from `rng`."""
    if name not in PROFILES:
        raise ValueError(f"unknown policy profile {name!r}; "
                         f"known: {sorted(PROFILES)}")
    spec = PROFILES[name]
    return ScriptedProfile(name, env, rng, kwargs=spec.get("kwargs"),
                           noise=spec.get("noise", 0.0), inject=spec.get("inject"))


def policy_schedule(n, mix, rng):
    """Turn shares into an exact assignment of `n` episodes to profiles.

    Largest-remainder apportionment, then shuffled. Sampling the profile
    independently per episode would be simpler but gets the composition of a
    small pilot wrong by a lot -- at n=12, a 5% share is present or absent
    depending on the seed, so the pilot would not be testing the mixture the
    manifest claims. Shuffling afterwards keeps the profile uncorrelated with
    the episode index (and so with the seed block), which matters because the
    splits are contiguous blocks.
    """
    names = sorted(mix)
    if n <= 0:
        return []
    exact = np.array([mix[k] * n for k in names], dtype=float)
    counts = np.floor(exact).astype(int)
    short = n - int(counts.sum())
    if short > 0:
        # Ties broken by name order, so this is a pure function of (n, mix).
        for i in np.argsort(-(exact - counts), kind="stable")[:short]:
            counts[i] += 1
    out = [name for name, c in zip(names, counts) for _ in range(c)]
    rng.shuffle(out)
    return out
