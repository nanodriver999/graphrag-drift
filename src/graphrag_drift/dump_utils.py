from __future__ import annotations

import re

SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def quote_identifier(value: str) -> str:
    if SAFE_IDENTIFIER.fullmatch(value):
        return value
    return "`" + value.replace("`", "``") + "`"
