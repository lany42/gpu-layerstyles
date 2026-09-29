# color-matcher reference fixtures

These two RGB PNGs (481×361 pixels each) are copied byte-for-byte from
`tests/data/` in Christopher Hahne's **color-matcher 0.6.0** wheel, published
March 30, 2025. They are loaded unchanged at full resolution.

- [PyPI release and published hashes](https://pypi.org/project/color-matcher/0.6.0/#files)
- [Source wheel](https://files.pythonhosted.org/packages/a0/3a/f3c2c5012f59235ff5885db7cc75dc209eca90e42ae3728db56f8a9e28a4/color_matcher-0.6.0-py3-none-any.whl)
- [Algorithm source at GitHub v0.5.0](https://github.com/hahnec/color-matcher/blob/40ea94e6c36cd93d119e97875e80eb409ada1422/color_matcher/mvgd_matcher.py)

The wheel's MKL and analytical MVGD solvers are unchanged from the cited GitHub
revision. Its solver file adds an unrelated `w2_img_dist` helper.

## SHA-256 provenance

| Asset | SHA-256 |
| --- | --- |
| `color_matcher-0.6.0-py3-none-any.whl` | `fd642f9414c33b7f3ebc96fe0888c1c6200836142664589ce2ccb52ebcda7734` |
| `scotland_house.png` | `ee72831c8230b569285b5254db2373e88e0befae74acb6022d42a0c8e91ad7d1` |
| `scotland_plain.png` | `459b97bafb1489f9ea2f1c376988a571b37cb56d46ea08a5ad91a98cb02cc177` |
| Wheel `tests/unit_test.py` | `faf55ead7944bac417063d2573887f5d59f5b6422c09f0eabf3d8aa915dd9c65` |
| Wheel `color_matcher/mvgd_matcher.py` | `d3f2367fa251ac08b9c12a40fc0dfe582a5ddd2682b293ba3fe3416853ca8853` |
| Wheel `color_matcher-0.6.0.dist-info/LICENSE` | `3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986` |

## Attribution and license

Original algorithm source: Copyright (c) 2020 Christopher Hahne
<info@christopherhahne.de>. Original upstream tests: Copyright (c) 2020 Christopher
Hahne <inbox@christopherhahne.de>. Their notices permit GPL version 3 or any later
version. The wheel metadata describes the package as GNU GPL V3.0.

The two photographs are unchanged upstream assets and retain their applicable
licensing. The wheel supplies no separate fixture copyright or license notices;
their provenance is that upstream test-data bundle. The
[original upstream GPL text](../../../LICENSES/color-matcher-LICENSE) is retained
unchanged.

The Python implementation and tests in GPU LayerStyles are copyright © 2026 Lany
Atwood <lany@colorized.life>, licensed under [AGPL-3.0-only](../../../LICENSE).
color-matcher served as the original algorithm source and numerical reference.
See [COPYRIGHT](../../../COPYRIGHT) for the project and original-source attribution.

`tests/test_color_match_reference.py` uses the house and landscape photographs
from upstream's test suite in two transfers: house to landscape at full strength,
and landscape to house at half strength. Both MKL and analytical MVGD are checked.
MKL is also checked from the house to a 6×7-pixel landscape crop, covering a
reference at a different resolution from the target.

Both implementations receive the same ordinary interleaved float32 `uint8 / 255`
inputs, with the method selected in the `ColorMatcher` constructor. Tests compare
the final blended/clipped node output directly against the library.
The absolute tolerance is `0.0005` per channel (`rtol=0`), about `0.13` on a 0–255
scale, allowing minor numerical differences in these practical comparisons.

The project's implementation uses PyTorch float32, batched explicit channel arithmetic,
stable centering and a fixed rank tolerance, reference reuse, and strength blending.
Neither NumPy nor color-matcher is imported by the runtime implementation.
