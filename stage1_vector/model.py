"""VectorJEPA: the official LeWM model with a vector encoder instead of a ViT.

`jepa.JEPA.encode` (le-wm/jepa.py:29) is hardcoded to pixel input: it reads
`info['pixels']`, runs a HuggingFace ViT, and takes the CLS token from
`last_hidden_state`. There is no vector-observation path anywhere in the
official repo -- the paper's experiments are all pixel-based. Everything
*downstream* of the encoder (`predict`, `rollout`, `criterion`, `get_cost`,
the autoregressive predictor, the action embedder) only ever touches
embeddings, so it is observation-agnostic already and needs no changes.

`VectorJEPA` overrides only `encode`: same CLS-token-style contract (one
embedding vector per timestep), but produced by `module.MLP` -- already
defined in their repo for exactly this kind of input/output-dim-to-dim
mapping -- reading the `state` column instead of `pixels`.
"""

from lewm.paths import add_lewm_official_to_path

add_lewm_official_to_path()

from einops import rearrange  # noqa: E402
from jepa import JEPA  # noqa: E402
from module import MLP, ARPredictor, Embedder  # noqa: E402


class VectorJEPA(JEPA):
    def encode(self, info):
        """Encode vector observations and actions into embeddings.

        info: dict with 'state' (B, T, D) and 'action' (B, T, A) keys.
        """
        state = info["state"].float()
        b = state.size(0)
        state = rearrange(state, "b t d -> (b t) d")
        emb = self.encoder(state)
        emb = self.projector(emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info


def build_model(
    state_dim,
    action_dim,
    *,
    embed_dim=128,
    history_size=3,
    encoder_hidden=256,
    predictor_depth=4,
    predictor_heads=8,
    predictor_dim_head=32,
    predictor_mlp_dim=512,
    action_smoothed_dim=32,
):
    """Build a VectorJEPA sized for a `state_dim`-D vector observation.

    Mirrors `config/train/model/lewm.yaml` in the official repo, with the
    ViT `encoder` swapped for an MLP and dimensions scaled down to match a
    73-D vector input instead of a 224x224 image -- the ViT-tiny config's
    hidden sizes (depth 6, 16 heads, mlp_dim 2048) are sized for a much
    higher-capacity pixel encoder than this baseline needs.
    """
    encoder = MLP(input_dim=state_dim, hidden_dim=encoder_hidden, output_dim=embed_dim)
    action_encoder = Embedder(
        input_dim=action_dim, smoothed_dim=action_smoothed_dim, emb_dim=embed_dim)
    predictor = ARPredictor(
        num_frames=history_size,
        input_dim=embed_dim,
        hidden_dim=embed_dim,
        output_dim=embed_dim,
        depth=predictor_depth,
        heads=predictor_heads,
        dim_head=predictor_dim_head,
        mlp_dim=predictor_mlp_dim,
    )
    projector = MLP(input_dim=embed_dim, hidden_dim=encoder_hidden, output_dim=embed_dim)
    pred_proj = MLP(input_dim=embed_dim, hidden_dim=encoder_hidden, output_dim=embed_dim)
    return VectorJEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
    )
