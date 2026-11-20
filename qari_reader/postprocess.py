"""Cleaning up model output."""

from __future__ import annotations

import re
import unicodedata

TASHKEEL = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")
TATWEEL = "ـ"


def strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def collapse_repeats(text: str, max_repeats: int = 2) -> tuple[str, bool]:
    """Small vision models sometimes get stuck repeating a line or phrase.

    Collapses a line that repeats more than `max_repeats` times in a row, and a
    run of the same phrase (3+ words) repeated 4+ times within a line.
    Returns (clean_text, changed).
    """
    changed = False
    out: list[str] = []
    run = 0
    for line in text.split("\n"):
        if out and line.strip() and line.strip() == out[-1].strip():
            run += 1
            if run >= max_repeats:
                changed = True
                continue
        else:
            run = 0
        out.append(line)
    text2 = "\n".join(out)

    phrase = re.compile(r"((?:\S+\s+){3,12}?\S+)(?:\s+\1){3,}")
    text3 = phrase.sub(r"\1", text2)
    if text3 != text2:
        changed = True
    return text3, changed


def normalise(text: str, keep_tashkeel: bool = True, remove_tatweel: bool = False) -> str:
    text = unicodedata.normalize("NFC", text)
    if not keep_tashkeel:
        text = TASHKEEL.sub("", text)
    if remove_tatweel:
        text = text.replace(TATWEEL, "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean(text: str, keep_tashkeel: bool = True, remove_tatweel: bool = False) -> tuple[str, list[str]]:
    """Full clean-up. Returns (text, warnings)."""
    warnings: list[str] = []
    text = strip_fences(text)
    text, looped = collapse_repeats(text)
    if looped:
        warnings.append("model repeated itself; repeats were removed — check this page")
    text = normalise(text, keep_tashkeel, remove_tatweel)
    return text, warnings
