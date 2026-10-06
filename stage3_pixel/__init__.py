"""Stage 3 -- pixel observations (brief section 1, "the actual visual LeWM
experiment").

A sibling package to `lewm/` and `stage1_vector/`, same convention as
before. Unlike Stage 1, no fork is needed here: `jepa.JEPA.encode` is
pixel-native already (it was Stage 1's vector input that needed the
override). This package only swaps the ViT's `image_size`/`patch_size` to
match our 64x64 renders instead of their 224x224 default -- `size="tiny"`
(hidden dim, depth, heads) stays theirs, unmodified.
"""
