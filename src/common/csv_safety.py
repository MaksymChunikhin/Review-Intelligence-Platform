"""Protect human-facing CSV exports from spreadsheet formula execution."""

from __future__ import annotations

from typing import Any

import pandas as pd


SPREADSHEET_FORMULA_PREFIXES = ("=", "+", "-", "@")


def escape_spreadsheet_formula(value: Any) -> Any:
    """Prefix potentially executable text while leaving non-strings unchanged."""
    if not isinstance(value, str) or not value:
        return value
    stripped = value.lstrip(" \t\r\n")
    if stripped.startswith(SPREADSHEET_FORMULA_PREFIXES) or value[0] in "\t\r":
        return "'" + value
    return value


def escape_dataframe_for_spreadsheet(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy whose text cells are safe to open in spreadsheet software."""
    result = frame.copy()
    for column in result.columns:
        if pd.api.types.is_object_dtype(result[column].dtype) or isinstance(
            result[column].dtype, pd.StringDtype
        ):
            result[column] = result[column].map(escape_spreadsheet_formula)
    return result
