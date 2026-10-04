"""
Locating the simulator this package collects from.

`lewm` lives in its own repository and `Car-Navigation-Env` lives in its own,
side by side, with the dependency pointing one way: we import the env, the env
has never heard of us. That is the same arrangement `dreamer-car-nav` uses for
`dreamerv3` (`scripts/_dreamer_common.py` inserts the sibling checkout on
`sys.path`), and it is the reason the env repo stayed untouched through this
work -- there is no shared package to edit.

Resolution order, first hit wins:

1. `CARNAV_ROOT`, if set -- for a checkout somewhere else entirely.
2. an already-importable `carnav` -- an editable install or an existing
   `PYTHONPATH` takes precedence over anything we would guess.
3. `../Car-Navigation-Env` relative to this repository.

Why a `sys.path` insert rather than requiring `pip install -e`: the env's
INTEGRATION.md documents a `PYTHONPATH` route precisely because its top-level
packages are named `env`, `render`, `agents` -- generic enough that installing
them into a shared site-packages is a bad idea. Doing it here means `python -m
lewm pilot` works in a fresh clone with nothing but numpy, which is also what
makes the pilot reproducible by someone else.
"""

import os
import sys

ENV_DIR_NAME = "Car-Navigation-Env"
_MARKERS = ("carnav.py", "env", "render", "baselines")

LEWM_OFFICIAL_DIR_NAME = "le-wm"
_OFFICIAL_MARKERS = ("jepa.py", "module.py", "train.py")


def _looks_like_carnav(root):
    return all(os.path.exists(os.path.join(root, m)) for m in _MARKERS)


def carnav_root():
    """Absolute path to the `Car-Navigation-Env` checkout, or None if unknown."""
    override = os.environ.get("CARNAV_ROOT")
    if override:
        root = os.path.abspath(os.path.expanduser(override))
        if not _looks_like_carnav(root):
            raise RuntimeError(
                f"CARNAV_ROOT={override!r} does not look like a "
                f"{ENV_DIR_NAME} checkout (expected {', '.join(_MARKERS)})")
        return root

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sibling = os.path.join(os.path.dirname(here), ENV_DIR_NAME)
    return sibling if _looks_like_carnav(sibling) else None


def add_carnav_to_path():
    """Put the simulator on `sys.path`. Idempotent; returns the path used.

    Called once at `lewm/__init__.py` import time, so every entry point --
    `python -m lewm`, `import lewm`, the test suite -- gets it for free and
    none of them has to repeat the dance.
    """
    try:                                    # already importable: leave it alone
        import carnav                       # noqa: F401
        return os.path.dirname(os.path.abspath(carnav.__file__))
    except ImportError:
        pass

    root = carnav_root()
    if root is None:
        raise ImportError(
            f"cannot find the {ENV_DIR_NAME} simulator. Expected it next to "
            f"this repository, or set CARNAV_ROOT=/path/to/{ENV_DIR_NAME}, or "
            f"put it on PYTHONPATH. `lewm` records trajectories from that "
            f"environment and does nothing without it.")
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def _looks_like_lewm_official(root):
    return all(os.path.exists(os.path.join(root, m)) for m in _OFFICIAL_MARKERS)


def lewm_official_root():
    """Absolute path to the `le-wm` (Maes et al.) checkout, or None if unknown.

    Same sibling-checkout arrangement as `carnav_root()`, for the same reason:
    this is someone else's repository (https://github.com/lucas-maes/le-wm),
    not a package, and its top-level module names (`jepa`, `module`,
    `train`) are generic enough that a shared site-packages install would be
    a trap. We import its `JEPA`/`module` classes unmodified rather than
    reimplementing the paper; only `stage1_vector/` and later stages need it,
    so unlike `carnav_root()` this is not bootstrapped at package import time.
    """
    override = os.environ.get("LEWM_OFFICIAL_ROOT")
    if override:
        root = os.path.abspath(os.path.expanduser(override))
        if not _looks_like_lewm_official(root):
            raise RuntimeError(
                f"LEWM_OFFICIAL_ROOT={override!r} does not look like a "
                f"{LEWM_OFFICIAL_DIR_NAME} checkout (expected "
                f"{', '.join(_OFFICIAL_MARKERS)})")
        return root

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sibling = os.path.join(os.path.dirname(here), LEWM_OFFICIAL_DIR_NAME)
    return sibling if _looks_like_lewm_official(sibling) else None


def add_lewm_official_to_path():
    """Put the official `le-wm` checkout on `sys.path`. Idempotent.

    Call this before `import jepa` / `import module` from that repository.
    """
    try:
        import jepa  # noqa: F401
        return os.path.dirname(os.path.abspath(jepa.__file__))
    except ImportError:
        pass

    root = lewm_official_root()
    if root is None:
        raise ImportError(
            f"cannot find the {LEWM_OFFICIAL_DIR_NAME} checkout (official "
            f"LeWM code, github.com/lucas-maes/le-wm). Expected it next to "
            f"this repository, or set LEWM_OFFICIAL_ROOT=/path/to/{LEWM_OFFICIAL_DIR_NAME}.")
    if root not in sys.path:
        sys.path.insert(0, root)
    return root
