"""build_model(): the unmodified official JEPA, sized to our 64x64 renders.

No subclass needed here, unlike `stage1_vector.model.VectorJEPA`. The ViT
patch/image size is the only thing that differs from
`config/train/model/lewm.yaml` -- `image_size=64, patch_size=8` (65 tokens:
8x8 patches + CLS) instead of their `image_size=224, patch_size=14` (257
tokens), because our scene is a low-poly procedural city, not a natural
image; matching their literal resolution would mean upsampling 64px source
frames to 224px and spending 4x the attention compute on interpolated
detail that doesn't exist. Everything else -- predictor, action encoder,
projector sizes -- is copied from their config as-is.
"""

from lewm.paths import add_lewm_official_to_path

add_lewm_official_to_path()

from jepa import JEPA  # noqa: E402
from module import MLP, ARPredictor, Embedder  # noqa: E402
from stable_pretraining.backbone.utils import vit_hf  # noqa: E402


# vit_hf's own size presets (hidden_size, depth, heads) -- embed_dim must
# match hidden_size exactly, since the CLS token is used as the embedding
# with no projection in between (JEPA.encode's self.projector maps
# embed_dim->embed_dim, it doesn't change dimensionality).
VIT_HIDDEN_SIZE = {"tiny": 192, "small": 384, "base": 768, "large": 1024}


def build_model(
    action_dim,
    *,
    image_size=64,
    patch_size=8,
    vit_size="tiny",
    embed_dim=None,  # defaults to vit_size's hidden_size if not given
    history_size=3,
    predictor_depth=6,
    predictor_heads=16,
    predictor_dim_head=64,
    predictor_mlp_dim=2048,
    action_smoothed_dim=32,
):
    embed_dim = embed_dim or VIT_HIDDEN_SIZE[vit_size]
    assert embed_dim == VIT_HIDDEN_SIZE[vit_size], (
        f"embed_dim={embed_dim} must equal vit_hf('{vit_size}')'s hidden_size "
        f"={VIT_HIDDEN_SIZE[vit_size]} -- it's the CLS token's own width.")
    encoder = vit_hf(
        size=vit_size, patch_size=patch_size, image_size=image_size,
        pretrained=False, use_mask_token=False,
    )
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
    # LayerNorm (module.MLP's default), not the official config's
    # BatchNorm1d -- same deliberate deviation as stage1_vector/model.py,
    # for the same reason: BatchNorm1d is unstable at the smaller batch
    # sizes pixel training needs here.
    projector = MLP(input_dim=embed_dim, hidden_dim=2048, output_dim=embed_dim)
    pred_proj = MLP(input_dim=embed_dim, hidden_dim=2048, output_dim=embed_dim)
    return JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
    )
