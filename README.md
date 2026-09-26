# GPU LayerStyles

The canonical home of this repository is at https://git.colorized.life/gpu-layerstyles/

Ten ComfyUI nodes for GPU-accelerated, float32 color adjustments, matching,
and image downscaling, plus image batch selection, concatenation, and transitions.

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://git.colorized.life/gpu-layerstyles.git gpu-layerstyles
# Restart ComfyUI.
```

## Nodes

| Display name | Node ID | Inputs and controls, in order | Outputs and behavior |
| --- | --- | --- | --- |
| GPU LayerStyles ColorBalance | `GPULayerStyles_ColorBalance` | `image`, `cyan_red`, `magenta_green`, `yellow_blue`, execution controls | `image`: balance RGB while preserving luminosity. |
| GPU LayerStyles Brightness Contrast V2 | `GPULayerStyles_BrightnessContrastV2` | `image`, `brightness`, `contrast`, `saturation`, execution controls | `image`: adjust brightness, per-frame contrast, then saturation. |
| GPU LayerStyles ColorTemperature | `GPULayerStyles_ColorTemperature` | `image`, `temperature`, execution controls | `image`: negative values warm; positive values cool. |
| GPU LayerStyles ColorMatch | `GPULayerStyles_ColorMatch` | `image`, `image_ref`, `method`, `strength`, execution controls | `image`: match RGB to the reference using MKL or MVGD; preserve target alpha. |
| GPU LayerStyles ImageScaleDown | `GPULayerStyles_ImageScaleDown` | `image`, `width`, `height`, `method`, execution controls | `image`: downscale to exact dimensions with bicubic or Lanczos-3 filtering. |
| GPU LayerStyles SliceImageBatch | `GPULayerStyles_SliceImageBatch` | `image`, `slice` | `image`: select frames with an index or `START:END[:STRIDE]`; negative indices and reverse strides are supported. |
| GPU LayerStyles BatchConcat | `GPULayerStyles_BatchConcat` | `images`: growing sockets `image_1` through `image_100` | `image`: concatenate connected images or batches in socket order; dimensions, dtype, and device must match. |
| GPU LayerStyles CrossFade | `GPULayerStyles_CrossFade` | `images_1`, `images_2`, `start_index`, `frames`, execution controls | `image`: keep the first batch's prefix, fade into the second batch over `frames`, then keep the rest of the second batch. |
| GPU LayerStyles TwoBatchLoop | `GPULayerStyles_TwoBatchLoop` | `images_1`, `images_2`, `blend_target`, `append_first_frame`, execution controls | `images`: crossfade both batch boundaries into a loop; optionally append its first frame. Each input needs at least `2 * blend_target + 1` frames. |
| GPU LayerStyles TwoBatchBridge | `GPULayerStyles_TwoBatchBridge` | `images`, `blend_target` | `bridge_first`: last `blend_target` frames; `bridge_last`: first `blend_target` frames; `first/last`: first frame of `bridge_first`, then last frame of `bridge_last`. |

Execution controls are `output_device`, then `batch_size`. All outputs are IMAGE
batches; multiple outputs are listed in socket order.

## Performance

**42.11× faster with GPU output; 6.12× with CPU offloading** in an informal,
user-reported benchmark against ComfyUI_LayerStyle: a **720×1280 video with
525 frames**, processed through ColorTemperature → ColorBalance → Brightness
Contrast. Times cover the complete chain, with the execution cache cleared
before each run. Hardware and repeat counts were not reported.

| Pipeline | Total time (ms) | Elapsed time reduction vs. upstream | Speedup vs. upstream |
| --- | ---: | ---: | ---: |
| GPU LayerStyles: CPU offloading | 11,157 | 83.7% | 6.12× |
| GPU LayerStyles: GPU output | 1,621 | 97.6% | 42.11× |
| ComfyUI_LayerStyle | 68,255 | 0.0% | 1.00× |

## License

Copyright © 2026 Lany Atwood <lany@colorized.life>. The project's Python source,
including the native PyTorch ColorMatch implementation and tests, is licensed
under [AGPL-3.0-only](LICENSE).

Christopher Hahne's color-matcher served as the original source for the MKL and
analytical MVGD algorithms and supplies the numerical reference used by our tests.
See [COPYRIGHT](COPYRIGHT) for project and original-source attribution.

ImageScaleDown's Lanczos algorithm is adapted from Pillow. Its
[MIT-CMU notices](LICENSES/Pillow-LICENSE) and source attribution ship in source
and wheel distributions alongside the project license.

The two vendored test photographs retain their upstream provenance and applicable
licensing; see the [fixture notices](tests/data/color_matcher/README.md). The
[original upstream GPL text](LICENSES/color-matcher-LICENSE) and attribution
notices ship alongside the project license in source and wheel distributions.
