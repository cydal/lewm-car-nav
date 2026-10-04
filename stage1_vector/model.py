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
from torch import nn  # noqa: E402


class VectorEncoder(nn.Module):
    """Stack of `module.MLP` blocks over the vector observation.

    The ViT this replaces spends essentially all of its 12 layers on a
    problem a vector observation does not have: discovering which pixels
    correspond to which object, from a ~150K-dim unstructured input,
    trained from scratch (`pretrained=false`). Our 73-D observation is
    already feature-engineered -- lidar/dynamics/nav/traffic blocks, each
    dimension physically meaningful on its own -- so there is no spatial
    structure left to discover. What a deeper encoder *can* still buy here
    is combining features across blocks (e.g. a lidar beam at the heading
    the car is turning toward, combined with velocity, is closer to
    "time to collision" than either alone); one `MLP` block has only one
    nonlinearity to do that with. `depth` stacked blocks with residual
    connections gives that some headroom without pretending we need
    anything like 12 layers of attention. Default depth is 2 -- a modest,
    swept hyperparameter, not a claim that it's exactly enough.
    """

    def __init__(self, input_dim, hidden_dim, embed_dim, depth=2):
        super().__init__()
        self.input_proj = MLP(input_dim, hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, hidden_dim) for _ in range(depth - 1)
        ])
        self.output_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = x + block(x)
        return self.output_proj(x)


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
    encoder_depth=2,
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
    encoder = VectorEncoder(
        input_dim=state_dim, hidden_dim=encoder_hidden, embed_dim=embed_dim, depth=encoder_depth)
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
