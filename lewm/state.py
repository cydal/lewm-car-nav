"""
Privileged simulator ground truth, recorded for *evaluation only*.

This is the half of the dataset the final pixel model must never see (brief
sections 5, 6 and 24). It exists so that a latent learned from RGB + action can
afterwards be probed for position, heading, velocity, goal distance and
obstacle distance, using numbers the model was never given.

Two rules shape what is in here:

**Only quantities the simulator actually has.** Every column below is read
straight off `env.car`, `env.targets`, `env.traffic`, `env.street`,
`env.traffic_lights` or `env.lidar`. Nothing is estimated, smoothed or
back-differenced. A quantity whose subsystem is switched off is *absent* --
there are no pedestrian columns in a dataset collected without pedestrians --
rather than present and filled with a placeholder, because a probe trained
against a constant column reports a meaningless R^2 instead of failing.

**The column set is fixed at construction and asserted every row.** The names
live in the manifest and in each episode file; a row that came out the wrong
width is a corrupted dataset, so it raises here rather than being written.

Sentinels: a distance/bearing column for an object that does not exist *at this
timestep* (no vehicle in the population, no signal on the map) is `-1.0`. That
is outside the range of every real value in those columns, and it is only
reachable when the subsystem is on but momentarily has nothing to report.
Filter on `dist >= 0` before probing.

The `info` dict already exposes a handful of these (`x`, `y`, `heading`,
`speed`, `steer_angle`, `dist_to_target`) and INTEGRATION.md labels them
"privileged ground truth, for logging and diagnostics only". This module is
that same idea, widened to the full set a probing experiment needs, and kept
structurally separate from the observations in the stored file so that handing
it to a model by accident takes a deliberate line of code.
"""

import numpy as np

NONE = -1.0     # "no such object right now"; see the module docstring

# Signal state encoding for the nearest-signal column, on the ego's approach axis.
SIGNAL_CODES = {"red": 0.0, "yellow": 1.0, "green": 2.0}

# Per-vehicle / per-pedestrian / per-signal column layouts of the auxiliary
# ground-truth blocks. World frame, metres, radians, m/s.
TRAFFIC_STATE_COLS = ("x", "y", "heading", "speed", "parked", "valid")
PEDESTRIAN_STATE_COLS = ("x", "y", "vx", "vy", "valid")
SIGNAL_STATE_COLS = ("phase", "timer", "steps_remaining")


def _bearing(car, tx, ty):
    """Body-frame bearing from the car to a world point, radians in [-pi, pi]."""
    b = np.arctan2(ty - car.y, tx - car.x) - car.heading
    return float(np.arctan2(np.sin(b), np.cos(b)))


class PrivilegedRecorder:
    """Builds one fixed-width row of ground truth per observation.

    Constructed once per collection run against the env it will read, so the
    column set is decided by that env's configuration rather than guessed per
    call.
    """

    def __init__(self, env, n_traffic_state=8, n_pedestrian_state=4,
                 traffic_state=True, signal_state=True, pedestrian_state=True):
        self.env = env
        cfg = env.cfg
        # The LIDAR-derived obstacle distance is only real if something asked
        # for a scan this step, and the only thing that does is the vector
        # observation. On an image-only env `lidar.last_distances` would still
        # hold its initialisation value, so the column is dropped rather than
        # recorded as a constant. Scanning here instead would be a second,
        # hidden draw on the LIDAR's RNG stream whenever `lidar_noise > 0`.
        self.has_lidar = getattr(env, "obs_type", "vector") in ("vector", "both")
        self.has_traffic = bool(cfg.traffic)
        self.has_signals = bool(cfg.traffic_lights)
        self.has_pedestrians = bool(cfg.pedestrians)
        self.has_signs = bool(cfg.speed_signs)

        self.n_traffic_state = int(n_traffic_state) if (traffic_state and self.has_traffic) else 0
        self.n_pedestrian_state = (int(n_pedestrian_state)
                                   if (pedestrian_state and self.has_pedestrians) else 0)
        self.want_signal_state = bool(signal_state and self.has_signals)

        self.names = self._build_names()

    # ------------------------------------------------------------------
    def _build_names(self):
        names = [
            # ego pose and motion, world frame
            "x", "y", "heading", "speed", "vx", "vy",
            "accel", "yaw_rate", "steer_angle", "slip",
            # task
            "target_idx", "dist_to_goal", "goal_x", "goal_y", "goal_bearing",
        ]
        if self.has_lidar:
            names.append("dist_to_obstacle")        # nearest LIDAR return, metres
        if self.has_traffic:
            names += ["dist_to_vehicle", "bearing_to_vehicle", "vehicle_speed",
                      "dist_to_moving_vehicle", "bearing_to_moving_vehicle",
                      "moving_vehicle_speed"]
        if self.has_signals:
            names += ["dist_to_signal", "signal_state", "signal_steps_remaining",
                      "red_light_violations"]
        if self.has_pedestrians:
            names += ["dist_to_pedestrian", "bearing_to_pedestrian",
                      "pedestrian_speed", "pedestrians_on_road"]
        if self.has_signs:
            names.append("speed_limit")
        return tuple(names)

    # ------------------------------------------------------------------
    def row(self):
        """Ground truth for the env's *current* state, as a float32 row."""
        env = self.env
        car = env.car
        vals = [
            car.x, car.y, car.heading, car.speed,
            # The bicycle model integrates position along heading + slip, so
            # these are the car's actual velocity components, not a projection
            # that ignores the slip angle.
            car.speed * np.cos(car.heading + car.slip),
            car.speed * np.sin(car.heading + car.slip),
            car.accel, car.yaw_rate, car.steer_angle, car.slip,
        ]

        if env.target_idx < len(env.targets):
            gx, gy = env.targets[env.target_idx]
            vals += [env.target_idx, float(np.hypot(gx - car.x, gy - car.y)),
                     gx, gy, _bearing(car, gx, gy)]
        else:
            # Every waypoint reached: the episode is over on this very step, so
            # there is no current goal. Distance 0 is the truth here (the env's
            # own `_dist_to_target` returns 0 too), and the goal position is the
            # last waypoint, which is where the car is.
            vals += [env.target_idx, 0.0, car.x, car.y, 0.0]

        if self.has_lidar:
            vals.append(float(env.lidar.last_distances.min()))

        if self.has_traffic:
            vals += self._traffic_scalars()
        if self.has_signals:
            vals += self._signal_scalars()
        if self.has_pedestrians:
            vals += self._pedestrian_scalars()
        if self.has_signs:
            vals.append(float(env.street.limit_at(car.x, car.y)))

        row = np.asarray(vals, dtype=np.float32)
        if row.shape != (len(self.names),):
            raise RuntimeError(f"privileged row is {row.shape} but the schema "
                               f"declares {len(self.names)} columns")
        return row

    # ------------------------------------------------------------------
    def _traffic_scalars(self):
        t, car = self.env.traffic, self.env.car
        out = []
        for lo, hi in ((0, len(t.x)), (0, t.n_moving)):   # all vehicles, then moving only
            if hi <= lo:
                out += [NONE, NONE, NONE]
                continue
            dx, dy = t.x[lo:hi] - car.x, t.y[lo:hi] - car.y
            d = np.hypot(dx, dy)
            i = int(np.argmin(d))
            out += [float(d[i]), _bearing(car, t.x[lo + i], t.y[lo + i]),
                    float(t.speed[lo + i])]
        return out

    def _signal_scalars(self):
        env, car = self.env, self.env.car
        lights = env.traffic_lights
        if not lights:
            return [NONE, NONE, NONE, float(env.red_light_violations)]
        d = [np.hypot(tl.x - car.x, tl.y - car.y) for tl in lights]
        tl = lights[int(np.argmin(d))]
        # Distance to the stop line, matching what the observation reports, so
        # a probe target and the observed feature mean the same thing.
        to_line = max(0.0, float(min(d)) - env._zone_half)
        state = tl.state(env._approach_axis(tl, car.x, car.y))
        return [to_line, SIGNAL_CODES[state], float(tl.steps_remaining),
                float(env.red_light_violations)]

    def _pedestrian_scalars(self):
        s, car = self.env.street, self.env.car
        n = s.n_pedestrians
        if n == 0:
            return [NONE, NONE, NONE, 0.0]
        d = np.hypot(s.px - car.x, s.py - car.y)
        i = int(np.argmin(d))
        return [float(d[i]), _bearing(car, s.px[i], s.py[i]),
                float(np.hypot(s.pvx[i], s.pvy[i])),
                float(s.on_road().sum())]

    # ------------------------------------------------------------------
    # Auxiliary per-object blocks: variable-size populations, so each is a
    # fixed (K, F) array with a `valid` flag instead of a ragged record.
    # ------------------------------------------------------------------
    def traffic_state(self):
        """Nearest `n_traffic_state` vehicles: world pose, speed, parked, valid."""
        k = self.n_traffic_state
        out = np.zeros((k, len(TRAFFIC_STATE_COLS)), dtype=np.float32)
        if k == 0:
            return out
        t, car = self.env.traffic, self.env.car
        n = len(t.x)
        if n == 0:
            return out
        d = np.hypot(t.x - car.x, t.y - car.y)
        order = np.argsort(d)[:k]
        for j, i in enumerate(order):
            out[j] = (t.x[i], t.y[i], t.heading[i], t.speed[i],
                      float(t.parked[i]), 1.0)
        return out

    def pedestrian_state(self):
        """Nearest `n_pedestrian_state` pedestrians: world position, velocity, valid."""
        k = self.n_pedestrian_state
        out = np.zeros((k, len(PEDESTRIAN_STATE_COLS)), dtype=np.float32)
        if k == 0:
            return out
        s, car = self.env.street, self.env.car
        if s.n_pedestrians == 0:
            return out
        d = np.hypot(s.px - car.x, s.py - car.y)
        for j, i in enumerate(np.argsort(d)[:k]):
            out[j] = (s.px[i], s.py[i], s.pvx[i], s.pvy[i], 1.0)
        return out

    def signal_state(self):
        """Every signal's phase, timer and steps-to-change, in `city.signals` order.

        Positions are fixed for the episode and live in the episode metadata
        (`signal_xy`), so only the time-varying part is stored per timestep.
        """
        if not self.want_signal_state:
            return np.zeros((0, len(SIGNAL_STATE_COLS)), dtype=np.float32)
        lights = self.env.traffic_lights
        out = np.zeros((len(lights), len(SIGNAL_STATE_COLS)), dtype=np.float32)
        for i, tl in enumerate(lights):
            out[i] = (tl.phase, tl.timer, tl.steps_remaining)
        return out

    def signal_xy(self):
        return np.asarray([(tl.x, tl.y) for tl in self.env.traffic_lights],
                          dtype=np.float32).reshape(-1, 2)
