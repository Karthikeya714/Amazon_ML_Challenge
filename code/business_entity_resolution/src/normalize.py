"""Country-agnostic normalization for business names and addresses.

Every rule here is a fixed, generic transformation (Unicode folding, a static
abbreviation dictionary, script transliteration) -- nothing looks up business
identity data, so this stays within the challenge's "no external lookup"
rule. The dictionaries are deliberately generic (not "if country == 'India'")
so the same code path handles the unseen France rows in test.

EDA findings this file is built to handle (see PLAN.md):
  - Random diacritics are injected into otherwise-Latin text as obfuscation
    noise (e.g. "Ássociates", "Sáaol", "TRÁNSALTA") -- NFKD + strip combining
    marks undoes this cleanly.
  - ~28% / ~18% of India business names in Source 2 / Source 3 are given in
    a native script (Devanagari, Kannada, ...) while Source 1 names are
    *always* Latin script -- these need script transliteration before any
    Latin-alphabet similarity feature will see any overlap at all.
  - India addresses never carry a PIN code (0% in this data); US addresses
    carry a 5-digit ZIP only ~10% of the time. Postcode features must
    degrade gracefully, not gate on postcode presence.
  - ~6% of Source 2/3 business names are website-domain-style strings
    ("healthwomensunited.com") with no spaces and reordered tokens.
  - Leading junk symbols ("-- ", "<< ") and bracketed suffixes ("[Inc]",
    "(LLC)") appear on otherwise-clean names.
  - House numbers sometimes carry leading zeros on one side only
    ("013614 Peacockfarm Rd" vs "13614 Peacockfarm Road").
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate as _indic_transliterate
    _HAS_INDIC = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_INDIC = False

# ---------------------------------------------------------------------------
# Unicode / script handling
# ---------------------------------------------------------------------------

# Unicode block -> indic_transliteration source-script constant. Ranges cover
# the common Indic scripts; anything outside these (and outside ASCII) is
# left alone rather than guessed at.
_SCRIPT_RANGES = []
if _HAS_INDIC:
    _SCRIPT_RANGES = [
        (0x0900, 0x097F, sanscript.DEVANAGARI),
        (0x0980, 0x09FF, sanscript.BENGALI),
        (0x0A00, 0x0A7F, sanscript.GURMUKHI),
        (0x0A80, 0x0AFF, sanscript.GUJARATI),
        (0x0B00, 0x0B7F, sanscript.ORIYA),
        (0x0B80, 0x0BFF, sanscript.TAMIL),
        (0x0C00, 0x0C7F, sanscript.TELUGU),
        (0x0C80, 0x0CFF, sanscript.KANNADA),
        (0x0D00, 0x0D7F, sanscript.MALAYALAM),
    ]


def _detect_script(s: str) -> Optional[str]:
    """Return the indic_transliteration source-script constant for the
    Unicode block that most of ``s``'s letters fall in, or None if the
    string is (mostly) Latin/ASCII already."""
    if not _HAS_INDIC:
        return None
    counts = {}
    for ch in s:
        cp = ord(ch)
        if cp < 0x0900:  # ASCII / Latin-1 -- not an Indic script
            continue
        for lo, hi, name in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[name] = counts.get(name, 0) + 1
                break
    if not counts:
        return None
    return max(counts, key=counts.get)


def transliterate_to_latin(s: str) -> str:
    """Romanize a native-script string (Devanagari, Kannada, ...) to a
    Latin-alphabet phonetic approximation. No-op for already-Latin text or
    when the optional dependency isn't available."""
    if not s:
        return s
    script = _detect_script(s)
    if script is None:
        return s
    try:
        return _indic_transliterate(s, script, sanscript.ITRANS)
    except Exception:
        return s


def strip_accents(s: str) -> str:
    """Fold accented Latin characters to their base letter (NFKD + drop
    combining marks). Handles both real diacritics and the random-accent
    obfuscation noise seen in this dataset."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------

# Ordered longest-first so e.g. "incorporated" matches before "inc".
_LEGAL_SUFFIXES = [
    "incorporated", "corporation", "corp", "inc",
    "private limited", "pvt ltd", "pvt", "private", "limited", "ltd",
    "llp", "llc", "plc", "co",
    "company", "enterprises", "enterprise",
    "gmbh", "sarl", "sasu", "sas", "eurl", "sci", "snc", "sa",
]
_LEGAL_SUFFIX_RE = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in _LEGAL_SUFFIXES) + r")\b\.?", re.IGNORECASE
)

_NAME_ABBREV = {
    "intl": "international", "int'l": "international",
    "mfg": "manufacturing", "mfr": "manufacturer",
    "svc": "service", "svcs": "services",
    "tech": "technologies", "techs": "technologies",
    "ent": "enterprises", "assoc": "associates", "assocs": "associates",
    "bros": "brothers", "dept": "department",
    "&": "and", "grp": "group", "mgmt": "management",
    "natl": "national", "dist": "district", "dba": "",
}

_LEADING_JUNK_RE = re.compile(r"^[\W_]+")
_BRACKETED_RE = re.compile(r"[\[\](){}]")
_MULTI_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_DOMAIN_TLD_RE = re.compile(r"\.(com|net|org|in|co|biz|info)\b", re.IGNORECASE)
_URL_SCHEME_RE = re.compile(r"^(https?://)?(www\.)?", re.IGNORECASE)


def _expand_tokens(tokens: List[str]) -> List[str]:
    out = []
    for t in tokens:
        rep = _NAME_ABBREV.get(t, t)
        if rep:
            out.extend(rep.split())
    return out


def is_domain_like(raw: str) -> bool:
    s = raw.strip()
    return bool(_DOMAIN_TLD_RE.search(s)) or (" " not in s and len(s) > 8 and any(c.isalpha() for c in s))


@dataclass
class NormalizedName:
    raw: str
    norm: str            # lowercase, accents stripped, suffixes/junk removed, abbrevs expanded
    core_tokens: List[str] = field(default_factory=list)
    sorted_core: str = ""       # tokens sorted alphabetically (word-order invariant)
    despaced: str = ""          # norm with spaces removed (for domain-style comparisons)
    legal_suffix: str = ""      # the suffix that was stripped, if any
    is_domain_like: bool = False


def normalize_name(raw: str) -> NormalizedName:
    raw = raw or ""
    s = transliterate_to_latin(raw)
    s = strip_accents(s)
    domain_like = is_domain_like(s)
    s = _URL_SCHEME_RE.sub("", s)
    s = _DOMAIN_TLD_RE.sub(" ", s)
    s = _BRACKETED_RE.sub(" ", s)
    s = _LEADING_JUNK_RE.sub("", s)
    s = s.lower()

    suffix_match = _LEGAL_SUFFIX_RE.search(s)
    legal_suffix = suffix_match.group(1) if suffix_match else ""
    s_wo_suffix = _LEGAL_SUFFIX_RE.sub(" ", s)

    s_wo_suffix = _NON_ALNUM_RE.sub(" ", s_wo_suffix)
    tokens = s_wo_suffix.split()
    tokens = _expand_tokens(tokens)
    norm = " ".join(tokens)
    sorted_core = " ".join(sorted(tokens))
    despaced = norm.replace(" ", "")

    return NormalizedName(
        raw=raw,
        norm=norm,
        core_tokens=tokens,
        sorted_core=sorted_core,
        despaced=despaced,
        legal_suffix=legal_suffix,
        is_domain_like=domain_like,
    )


# ---------------------------------------------------------------------------
# Address normalization
# ---------------------------------------------------------------------------

_STREET_ABBREV = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "ln": "lane", "dr": "drive",
    "ct": "court", "cir": "circle", "ter": "terrace", "pl": "place",
    "sq": "square", "hwy": "highway", "pkwy": "parkway", "apt": "apartment",
    "bldg": "building", "fl": "floor", "flr": "floor", "no": "number",
    "nr": "near", "opp": "opposite", "sec": "sector", "colony": "colony",
    "rte": "route", "jct": "junction", "expy": "expressway",
}

_LANDMARK_RE = re.compile(
    r"\b(near|nr\.?|opp\.?|opposite|behind|beside|adjacent to|next to)\b[^,]*", re.IGNORECASE
)
# US ZIP is ~10% present in this data; a genuine 6-digit Indian PIN is 0%
# present (EDA-verified), so a 6-digit fallback would only ever match
# zero-padded house/survey numbers (e.g. "013614 Peacockfarm Rd") -- the
# negative lookaround below requires an *exact* 5-digit run, so a 6-digit
# number never matches at all.
_POSTCODE_RE = re.compile(r"(?<!\d)\d{5}(?:-\d{4})?(?!\d)")
_HOUSE_NUM_RE = re.compile(r"\b0*(\d{2,}[a-zA-Z]?(?:/\d+[a-zA-Z]?)?)\b")


@dataclass
class NormalizedAddress:
    raw: str
    norm: str
    tokens: List[str] = field(default_factory=list)
    postcode: str = ""
    house_numbers: List[str] = field(default_factory=list)
    landmark: str = ""
    is_missing: bool = True


def normalize_address(raw: str, country: str = "") -> NormalizedAddress:
    raw = raw or ""
    if not raw.strip():
        return NormalizedAddress(raw=raw, norm="", is_missing=True)

    s = transliterate_to_latin(raw)
    s = strip_accents(s)

    landmark_match = _LANDMARK_RE.search(s)
    landmark = landmark_match.group(0).strip() if landmark_match else ""
    s_wo_landmark = _LANDMARK_RE.sub(" ", s)

    postcode_match = _POSTCODE_RE.search(s_wo_landmark)
    postcode = postcode_match.group(0) if postcode_match else ""

    house_numbers = sorted(
        set(m.group(1).lower() for m in _HOUSE_NUM_RE.finditer(s_wo_landmark)) - {postcode}
    )

    s_lower = s_wo_landmark.lower()
    s_lower = _NON_ALNUM_RE.sub(" ", s_lower)
    raw_tokens = s_lower.split()
    tokens = []
    for t in raw_tokens:
        tokens.append(_STREET_ABBREV.get(t, t))
    norm = " ".join(tokens)

    return NormalizedAddress(
        raw=raw,
        norm=norm,
        tokens=tokens,
        postcode=postcode,
        house_numbers=house_numbers,
        landmark=landmark,
        is_missing=False,
    )
