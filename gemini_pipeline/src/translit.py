"""Indic-script -> Latin transliteration.

Two layers:
1. A token dictionary *learned from the training ground truth*: for matched
   pairs where the candidate name is written in an Indic script and has the
   same number of tokens as the (Latin) Source-1 name, tokens are aligned by
   position and the most frequent Latin counterpart is kept.  This uses only
   the provided training data (no external resources).
2. A deterministic rule-based fallback for tokens not in the dictionary.  All
   major Indic Unicode blocks share the ISCII layout, so one offset table
   covers Devanagari, Bengali, Gurmukhi, Gujarati, Odia, Tamil, Telugu,
   Kannada and Malayalam.
"""
import re

import polars as pl

INDIC_RE = r"[ऀ-෿]"
_BLOCKS = [0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00]

_VOWELS = {0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri",
           0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au"}
_CONS = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh", 0x1C: "j",
         0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t",
         0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "f", 0x2C: "b",
         0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l",
         0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
         0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "d", 0x5D: "dh", 0x5E: "f", 0x5F: "y"}
_SIGNS = {0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x45: "e", 0x46: "e",
          0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au"}
_VIRAMA = 0x4D
_NASAL = {0x01: "n", 0x02: "n", 0x03: "h"}


def _offset(ch):
    cp = ord(ch)
    for b in _BLOCKS:
        if b <= cp < b + 0x80:
            return cp - b
    return None


def rule_translit(token: str) -> str:
    """Rule-based romanisation of one Indic token (schwa deletion at word end)."""
    out = []
    pending_a = False  # consonant emitted, inherent 'a' not yet resolved
    for ch in token:
        off = _offset(ch)
        if off is None:
            if pending_a:
                out.append("a")
                pending_a = False
            if ch.isascii() and ch.isalnum():
                out.append(ch.lower())
            continue
        if off in _CONS:
            if pending_a:
                out.append("a")
            out.append(_CONS[off])
            pending_a = True
        elif off in _SIGNS:
            out.append(_SIGNS[off])
            pending_a = False
        elif off == _VIRAMA:
            pending_a = False
        elif off in _VOWELS:
            if pending_a:
                out.append("a")
                pending_a = False
            out.append(_VOWELS[off])
        elif off in _NASAL:
            if pending_a:
                out.append("a")
                pending_a = False
            out.append(_NASAL[off])
        elif 0x66 <= off <= 0x6F:
            if pending_a:
                out.append("a")
                pending_a = False
            out.append(str(off - 0x66))
    s = "".join(out)
    s = re.sub(r"(.)\1+", r"\1", s)  # collapse doubled letters (aa->a, ii->i)
    return s


def learn_token_dict(pairs: pl.DataFrame, min_count: int = 2, min_share: float = 0.4) -> dict:
    """Learn native->latin token map from aligned pairs.

    pairs: columns `lat` (list[str], Source-1 tokens) and `nat` (list[str],
    candidate tokens, at least one Indic).  Only equal-length pairs are used.
    """
    p = (pairs.filter(pl.col("lat").list.len() == pl.col("nat").list.len())
         .select("lat", "nat").explode(["lat", "nat"])
         .filter(pl.col("nat").str.contains(INDIC_RE)))
    cnt = p.group_by("nat", "lat").len()
    tot = cnt.group_by("nat").agg(pl.col("len").sum().alias("tot"))
    best = (cnt.sort("len", descending=True).group_by("nat").first()
            .join(tot, on="nat")
            .filter((pl.col("len") >= min_count) & (pl.col("len") / pl.col("tot") >= min_share)))
    return dict(zip(best["nat"].to_list(), best["lat"].to_list()))
