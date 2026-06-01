"""Shared 12/16/4 season label maps (independent of gaussian_twelve classifier)."""

from __future__ import annotations

TWELVE_TO_FOUR: dict[str, str] = {
    "light_spring": "spring",
    "true_spring": "spring",
    "bright_spring": "spring",
    "light_summer": "summer",
    "true_summer": "summer",
    "soft_summer": "summer",
    "soft_autumn": "autumn",
    "true_autumn": "autumn",
    "deep_autumn": "autumn",
    "deep_winter": "winter",
    "true_winter": "winter",
    "bright_winter": "winter",
}

SIXTEEN_TO_TWELVE: dict[str, str] = {
    "true_bright": "bright_spring",
    "true_light": "light_summer",
    "true_soft": "soft_summer",
    "true_deep": "deep_winter",
}
