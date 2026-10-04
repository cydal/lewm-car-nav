"""Load the wandb API key from `le-wm/.env` without a dotenv dependency.

The key lives next to the official repo checkout (`lewm_official_root()`),
not in this repo, and is gitignored there. The file holds one line,
`WAND_API=...`; wandb itself looks for `WANDB_API_KEY`, so this maps the one
to the other rather than asking the user to rename it.
"""

import os

from lewm.paths import lewm_official_root


def load_wandb_key():
    """Set `WANDB_API_KEY` from `<le-wm>/.env` if present and not already set.

    Returns True if a key is set in the environment afterward, else False.
    """
    if os.environ.get("WANDB_API_KEY"):
        return True

    root = lewm_official_root()
    if root is None:
        return False

    env_path = os.path.join(root, ".env")
    if not os.path.exists(env_path):
        return False

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() == "WAND_API" and value.strip():
                os.environ["WANDB_API_KEY"] = value.strip()
                return True
    return False
