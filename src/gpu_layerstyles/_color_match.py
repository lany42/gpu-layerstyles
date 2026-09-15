# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Native batched float32 MKL and analytical MVGD for GPU LayerStyles.

Original algorithm source: Christopher Hahne's color-matcher 0.6.0,
color_matcher/mvgd_matcher.py, also in GitHub v0.5.0 at commit
40ea94e6c36cd93d119e97875e80eb409ada1422.
Original source copyright (c) 2020 Christopher Hahne <info@christopherhahne.de>.

This project's implementation uses native PyTorch, batched explicit RGB
arithmetic, stable centering, a fixed float32 rank tolerance, finite checks,
and reusable reference statistics. It performs no runtime image normalization.
See COPYRIGHT for project and original-source attribution.
"""

from dataclasses import dataclass

import torch

RANK_RTOL = 3 * torch.finfo(torch.float32).eps


@dataclass(frozen=True)
class PreparedReference:
    mean: torch.Tensor
    covariance: torch.Tensor | None = None
    p_z: torch.Tensor | None = None


def _check_result(tensor: torch.Tensor, stage: str) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"ColorMatch numerical failure: non-finite {stage}.")


def _statistics(
    rgb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not torch.isfinite(rgb).all():
        raise ValueError("ColorMatch active statistics require finite RGB values.")
    pixels = rgb.movedim(-1, 1).flatten(2).contiguous()
    # Mean-rounding must not create variance in constant images.
    first = pixels[..., :1]
    delta = pixels - first
    mean_delta = delta.mean(dim=-1, keepdim=True)
    mean = first + mean_delta
    centered = delta - mean_delta

    # Six independent sample-covariance entries, mirrored exactly. For N=1,
    # centered is zero and the denominator is one, so covariance is zero.
    denominator = max(pixels.shape[-1] - 1, 1)
    rr, rg, rb, gg, gb, bb = [
        (centered[:, i] * centered[:, j]).sum(dim=-1) / denominator
        for i, j in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    ]
    covariance = torch.stack((rr, rg, rb, rg, gg, gb, rb, gb, bb), dim=-1)
    covariance = covariance.reshape(-1, 3, 3)
    _check_result(mean, "channel means")
    _check_result(covariance, "covariance")
    return mean, centered, covariance


def _transform(matrix: torch.Tensor, pixels: torch.Tensor) -> torch.Tensor:
    """Multiply [B,3,3] by [B,3,N] without autocast/TF32 matmul rounding."""
    red, green, blue = pixels.unbind(dim=-2)
    return torch.stack(
        [
            matrix[:, i, 0, None] * red
            + matrix[:, i, 1, None] * green
            + matrix[:, i, 2, None] * blue
            for i in range(3)
        ],
        dim=-2,
    )


def _cross_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Multiply [B,3,N] by [B,N,3] using contiguous channel reductions."""
    return torch.stack(
        [(left[:, i] * right[:, j]).sum(dim=-1) for i in range(3) for j in range(3)],
        dim=-1,
    ).reshape(-1, 3, 3)


def _covariance_inverse(covariance: torch.Tensor) -> torch.Tensor:
    return torch.linalg.pinv(covariance, atol=0.0, rtol=RANK_RTOL, hermitian=True)


def _mkl_transfer(
    covariance: torch.Tensor, reference_covariance: torch.Tensor
) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(covariance)
    values = values.clamp_min(0)
    roots = values.sqrt()
    inverse = torch.where(
        values > values.amax(dim=-1, keepdim=True) * RANK_RTOL,
        1 / (roots + 2**-52),  # Upstream epsilon, only for retained modes.
        0,
    )
    factor = roots.unsqueeze(-1) * vectors.mT
    middle = _transform(_transform(factor, reference_covariance), factor.mT)
    middle = (middle + middle.mT) * 0.5
    _check_result(middle, "MKL intermediate covariance")
    middle_values, middle_vectors = torch.linalg.eigh(middle)
    middle_root = _transform(
        middle_vectors * middle_values.clamp_min(0).sqrt().unsqueeze(-2),
        middle_vectors.mT,
    )
    inverse_factor = vectors * inverse.unsqueeze(-2)
    return _transform(_transform(inverse_factor, middle_root), inverse_factor.mT)


@torch.no_grad()
def prepare_reference(rgb: torch.Tensor, method: str) -> PreparedReference:
    """Retain only mean/covariance for MKL, or mean/P_Z for analytical MVGD."""
    with torch.autocast(device_type=rgb.device.type, enabled=False):
        try:
            mean, centered, covariance = _statistics(rgb.float())
            if method == "mkl":
                return PreparedReference(mean=mean, covariance=covariance)
            # Preserve the upstream tall pseudoinverse and its rank truncation.
            tall = _transform(_covariance_inverse(covariance).mT, centered).mT
            _check_result(tall, "MVGD reference system")
            p_z = torch.linalg.pinv(tall, atol=0.0, rtol=RANK_RTOL).contiguous()
            _check_result(p_z, "MVGD reference pseudoinverse")
            return PreparedReference(mean=mean, p_z=p_z)
        except torch.linalg.LinAlgError as error:
            raise RuntimeError(
                f"ColorMatch {method} numerical failure preparing reference: {error}"
            ) from error


@torch.no_grad()
def match_rgb(
    rgb: torch.Tensor, reference: PreparedReference, method: str
) -> torch.Tensor:
    """Return raw matched RGB before strength blending and clipping."""
    with torch.autocast(device_type=rgb.device.type, enabled=False):
        try:
            _, centered, covariance = _statistics(rgb.float())
            if method == "mkl":
                transfer = _mkl_transfer(covariance, reference.covariance)
            else:
                transfer = _transform(
                    _cross_product(reference.p_z, centered),
                    _covariance_inverse(covariance),
                )
            matched = _transform(transfer, centered) + reference.mean
            _check_result(matched, f"{method} matched RGB")
            return matched.movedim(1, -1).reshape(rgb.shape)
        except torch.linalg.LinAlgError as error:
            raise RuntimeError(
                f"ColorMatch {method} numerical failure matching target: {error}"
            ) from error


def color_match(
    rgb: torch.Tensor, reference: PreparedReference, method: str, strength: float
) -> torch.Tensor:
    if strength == 0:
        return rgb
    matched = match_rgb(rgb, reference, method)
    blended = rgb + strength * (matched - rgb)
    _check_result(blended, "strength blend")
    return blended.clamp_(0, 1)
