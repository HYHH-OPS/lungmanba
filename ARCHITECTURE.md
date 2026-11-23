# Methods correspondence and provenance

The supplied revision specification is the authority for this reference implementation. A locally available earlier design note selected Stage-2; this release follows the explicit Stage-3 requirement instead. No final experimental training source was found in the searched local document, desktop, or download trees. Historical entry points reference missing model packages and are not a reproducible final implementation.

- `models/encoder_3d.py:Encoder3D`: five 3×3×3 stride convolutions, padding 1, InstanceNorm and LeakyReLU. Input B×1×32×128×128 gives Stage-3 B×96×16×16×16 and Stage-5 B×256×8×4×4. These are computed dimensions, not independently verified manuscript tensor annotations.
- `GLFPB`: global 4×4×4 adaptive pooling; local 2×5×10 crop from Stage-3. Coordinates use cumulative convolution stride (2,8,8) with floor rounding. Replicate padding is an explicit reference boundary policy; equivalence to final experiments is unconfirmed. Even crop dimensions place the center at index size//2. Both streams have independent projections.
- `BidirectionalMamba`: X=[G;L;queries], H=LayerNorm(X+W[Forward(X);reverse(Backward(reverse(X)))]). Independent Transformers Mamba mixers perform selective scans. Split H into V (164) and Q (32). Queries at the sequence tail do not give both causal directions full visual context.
- `LungMambaEAR`: decoder weights frozen, but autograd through the decoder remains active. Caption targets exclude visual positions and padded text. FAB pools V or Q and applies the same five linear heads and masked BCE. Each field loss is averaged over available entries; five losses are summed.
- `models/sfc.py`: inference-only section state, forced heading transitions, body length cap, repeated bigram suppression over 16 body tokens, and EOS transitions. Disabled mode permits unstructured greedy output. Forced sections do not establish factual accuracy.
- `training/train.py`: independently trains visual components and chooses checkpoints by token-weighted validation caption NLL. No original preprocessing parameters, field ontology, or original SFC thresholds are claimed.

## Verification limits

Synthetic tests check real Mamba scans at reduced width, geometry at the nominal input, both FAB gradient paths, frozen decoder parameters, missing labels and valid negatives, SFC toggle, and checkpoint loading. Full 2.8B loading, GPU BF16 training, original-paper metrics, and final experimental equivalence require separate validation. Legacy scripts and documents are archival only and may reference unavailable packages or obsolete evaluation protocols.
