"""Backward-compatible face parsing facade."""

from __future__ import annotations

from app.backends.parsing import ParsingResult, glasses_pixel_ratio, parse_face

__all__ = ["ParsingResult", "glasses_pixel_ratio", "parse_face"]
