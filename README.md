# GPU LayerStyles

The canonical home of this repository is at https://git.colorized.life/gpu-layerstyles/

Five ComfyUI nodes for GPU-accelerated, float32 color adjustments, matching,
and image downscaling.

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://git.colorized.life/gpu-layerstyles.git gpu-layerstyles
# Restart ComfyUI.
```

## Nodes

| Display name | Node ID | Adjustment controls, in order |
| --- | --- | --- |
| GPU LayerStyles ColorBalance | `GPULayerStyles_ColorBalance` | `cyan_red`, `magenta_green`, `yellow_blue` |
| GPU LayerStyles Brightness Contrast V2 | `GPULayerStyles_BrightnessContrastV2` | `brightness`, `contrast`, `saturation` |
| GPU LayerStyles ColorTemperature | `GPULayerStyles_ColorTemperature` | `temperature` |
| GPU LayerStyles ColorMatch | `GPULayerStyles_ColorMatch` | `image_ref`, `method`, `strength` |
| GPU LayerStyles ImageScaleDown | `GPULayerStyles_ImageScaleDown` | `width`, `height`, `method` |

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
