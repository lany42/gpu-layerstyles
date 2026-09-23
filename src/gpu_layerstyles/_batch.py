# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Strict Python-style batch selections shared by image batch nodes."""

import re


def parse_batch_selection(expression: str, length: int) -> range:
    if not isinstance(expression, str):
        raise TypeError("slice must be a string containing an index or slice.")

    parts = expression.split(":")
    if not expression.strip() or len(parts) > 3:
        raise ValueError(
            f"Invalid slice {expression!r}; expected an integer or START:END[:STRIDE]."
        )

    names = ("index",) if len(parts) == 1 else ("START", "END", "STRIDE")
    values = []
    for name, part in zip(names, parts):
        token = part.strip()
        if not token and len(parts) > 1:
            values.append(None)
            continue
        if re.fullmatch(r"[+-]?[0-9]+", token) is None:
            raise ValueError(
                f"Invalid {name} {token!r} in slice {expression!r}; "
                "expected a signed decimal integer."
            )
        value = int(token)
        if name != "STRIDE":
            upper = length if name == "END" else length - 1
            if not -length <= value <= upper:
                raise IndexError(
                    f"{name} {value} in slice {expression!r} is out of range for "
                    f"a batch of {length} images; expected {-length} <= {name} <= {upper}."
                )
        elif value == 0:
            raise ValueError(f"STRIDE must be nonzero in slice {expression!r}.")
        values.append(value)

    if len(values) == 1:
        index = range(length)[values[0]]
        return range(index, index + 1)

    # Keep omitted bounds as None: reverse slices treat them differently from -1.
    selection = range(length)[slice(*values)]
    if not selection:
        raise ValueError(
            f"Slice {expression!r} selects no images from a batch of {length} images; "
            "the selection must be nonempty."
        )
    return selection
