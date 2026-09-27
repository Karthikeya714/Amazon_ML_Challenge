"""Loading + normalisation of business names and addresses (vectorised polars).

Output columns per record:
    rid        int   row id inside its source table
    entity_id  str
    src        int   1/2/3
    country    str   (open set - never hard-coded)
    name_tok   list[str]  canonical name tokens (transliterated, abbreviations unified)
    core_tok   list[str]  name tokens without legal forms / titles / decorations
    name_str   str        " ".join(name_tok)
    core_str   str        " ".join(core_tok)
    concat     str        "".join(core_tok)  (matches domain-style names)
    is_domain  bool
    alias_str  str        part of the name after an alias marker (dba, t/a, formerly, nee, aka)
    addr_tok   list[str]  address word tokens (abbrev expanded, no state, no generic words)
    addr_num   list[str]  numbers in the address (leading zeros stripped)
    addr_str   str        cleaned address (words+numbers, in order)
    state      str        canonical state/region code when recognisable, else ""
    addr_missing bool
"""
import polars as pl

from translit import INDIC_RE, rule_translit

LATIN_MARKS = r"[̀-ͯ]"
TOKEN_SPLIT = r"[^\p{L}\p{M}\p{N}]+"

NAME_CANON = {
    "limited": "ltd", "private": "pvt", "incorporated": "inc", "corporation": "corp",
    "company": "co", "centre": "center", "service": "services", "cntr": "center",
    "intl": "international", "mfg": "manufacturing", "bros": "brothers", "assoc": "associates",
    "tech": "technologies", "technology": "technologies", "pvtltd": "pvt ltd",
}
NAME_STOP = {
    "ltd", "pvt", "inc", "corp", "co", "llc", "llp", "lp", "pc", "pllc", "pa", "plc", "public",
    "the", "and", "of", "dr", "mr", "mrs", "ms", "smt", "sri", "shri", "center", "services",
    "partners", "group", "board", "formerly", "dba", "aka", "as", "doing", "business", "known",
    "nee", "ta", "id", "www", "com", "net", "org", "sa", "sas", "sarl", "eurl", "sci", "snc",
    "ste", "societe", "cie", "et", "de", "du", "des", "la", "le", "les", "sasu", "gmbh", "sl",
    "trading", "a", "m", "s", "l", "c", "p", "d", "k", "b", "f", "t", "n", "g", "h", "j", "i",
    "e", "o", "r", "u", "v", "w", "x", "y", "z", "q",
}
ALIAS_RE = r"\b(?:formerly|dba|d/b/a|aka|a\.k\.a\.?|t/a|trading as|doing business as|nee|known as|fka|f/k/a)\b"

ADDR_CANON = {
    "st": "street", "str": "street", "rd": "road", "raod": "road", "dr": "drive", "ave": "avenue",
    "av": "avenue", "blvd": "boulevard", "bd": "boulevard", "bvd": "boulevard", "ln": "lane",
    "ct": "court", "cir": "circle", "pl": "place", "hwy": "highway", "pkwy": "parkway",
    "ter": "terrace", "trl": "trail", "sq": "square", "mt": "mount", "ft": "fort", "nr": "near",
    "opp": "opposite", "r": "rue", "fl": "floor", "flr": "floor", "apt": "apartment",
    "ste": "suite", "rte": "route", "chem": "chemin", "imp": "impasse", "fbg": "faubourg",
    "hn": "house", "hno": "house", "bldg": "building", "cr": "crescent", "jn": "junction",
    "mg": "mg", "pb": "pb", "sec": "sector", "extn": "extension", "ext": "extension",
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
    "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
}
ADDR_STOP = {
    "street", "road", "drive", "avenue", "lane", "court", "circle", "place", "no", "door", "flat",
    "plot", "unit", "apartment", "suite", "floor", "near", "opposite", "po", "box", "pmb", "city",
    "township", "cdp", "the", "of", "block", "house", "building", "rue", "de", "du", "la", "le",
    "des", "and", "boulevard", "null", "na", "none", "nos", "number", "at", "post", "dist",
    "district", "taluk", "tq", "village", "vill", "ward", "sy", "survey", "sector", "phase",
    "stage", "main", "cross", "highway", "way", "parkway", "terrace", "trail", "square",
    "route", "chemin", "impasse", "allee", "d", "l", "a", "b", "c", "e", "f", "g", "h", "i",
    "j", "k", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "new hampshire": "nh",
    "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn",
    "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
    "puerto rico": "pr",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp",
    "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl",
    "odisha": "od", "orissa": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk",
    "tamil nadu": "tn", "telangana": "tg", "tripura": "tr", "uttar pradesh": "up",
    "uttarakhand": "uk", "west bengal": "wb", "delhi": "dl", "jammu and kashmir": "jk",
    "jammu kashmir": "jk", "ladakh": "la", "chandigarh": "ch", "puducherry": "py",
    "pondicherry": "py", "andaman and nicobar islands": "an", "dadra and nagar haveli": "dn",
    "daman and diu": "dd", "lakshadweep": "ld",
}
IN_CODE_ALIASES = {"ts": "tg", "or": "od", "ct": "cg", "ut": "uk", "uttaranchal": "uk", "dl": "dl"}


def _state_map(country: str) -> dict:
    """Component string -> canonical code, for the two known countries (other countries: {})."""
    if country == "US":
        m = dict(US_STATES)
        m.update({v: v for v in US_STATES.values()})
        return m
    if country == "India":
        m = dict(IN_STATES)
        m.update({v: v for v in IN_STATES.values()})
        m.update(IN_CODE_ALIASES)
        return m
    return {}


def load_source(path, src: int) -> pl.DataFrame:
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    return (df.rename({"business_name": "name", "business_address": "addr"})
              .with_row_index("rid")
              .with_columns(pl.lit(src, pl.Int8).alias("src"),
                            pl.col("country").fill_null(""),
                            pl.col("rid").cast(pl.UInt32)))


def _base_clean(col: str) -> pl.Expr:
    return (pl.col(col).fill_null("").str.to_lowercase()
            .str.normalize("NFKD").str.replace_all(LATIN_MARKS, ""))


def _map_list(df: pl.DataFrame, col: str, mapping: dict, drop: set | None = None) -> pl.DataFrame:
    """Apply a token->token(s) map to a list[str] column (vectorised through explode)."""
    long = (df.select("rid", pl.col(col)).explode(col)
              .with_columns(pl.col(col).replace(mapping) if mapping else pl.col(col))
              .with_columns(pl.col(col).str.split(" ")).explode(col))
    long = long.filter(pl.col(col).is_not_null() & (pl.col(col) != ""))
    if drop:
        long = long.filter(~pl.col(col).is_in(list(drop)))
    agg = long.group_by("rid", maintain_order=True).agg(pl.col(col))
    return df.drop(col).join(agg, on="rid", how="left").with_columns(
        pl.col(col).fill_null(pl.lit([], pl.List(pl.Utf8))))


def translit_tokens(df: pl.DataFrame, col: str, tdict: dict) -> pl.DataFrame:
    """Replace Indic tokens by dictionary entry, else rule-based romanisation."""
    toks = df.select(pl.col(col).explode()).drop_nulls().unique()
    nat = toks.filter(pl.col(col).str.contains(INDIC_RE))[col].to_list()
    mapping = {t: tdict.get(t) or rule_translit(t) for t in nat}
    return _map_list(df, col, mapping)


def raw_name_tokens(col: str) -> pl.Expr:
    """Cleaned name tokens before transliteration / canonicalisation."""
    return (_base_clean(col)
            .str.replace_all(r"\(\s*id\s*:?\s*\d+\s*\)|\bid\s*:\s*\d+", " ")
            .str.replace_all(r"\d{7,}", " ")
            .str.replace_all(r"\bwww\.", " ")
            .str.replace_all(r"\.(?:com|net|org|co\.in|in|fr|biz|info|us|io|co)\b", " ")
            .str.replace_all(r"[&+]", " and ")
            .str.replace_all(r"['’`\.]", "")
            .str.replace_all(TOKEN_SPLIT, " ").str.strip_chars()
            .str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
            # "leetspeak" typos seen in the data (Br0thers, 8uilders, Kwa1ity, HOLDING5, denta1care):
            # inside tokens that mix letters and digits, map look-alike digits back to letters
            .list.eval(pl.when(pl.element().str.contains(r"\p{L}") & pl.element().str.contains(r"\d"))
                         .then(pl.element().str.replace_many(["0", "1", "3", "4", "5", "7", "8"],
                                                             ["o", "l", "e", "a", "s", "t", "b"]))
                         .otherwise(pl.element())))


def normalize_names(df: pl.DataFrame, tdict: dict) -> pl.DataFrame:
    s = _base_clean("name")
    df = df.with_columns(
        s.str.contains(r"[a-z0-9]\.(?:com|net|org|in|fr|biz|info|us|io|co)\b").alias("is_domain"),
        s.str.extract(ALIAS_RE + r"\s+(.+)$", 1).fill_null("").alias("alias_raw"),
        raw_name_tokens("name").alias("name_tok"),
    )
    df = translit_tokens(df, "name_tok", tdict)
    df = _map_list(df, "name_tok", NAME_CANON)
    core = pl.col("name_tok").list.eval(pl.element().filter(~pl.element().is_in(list(NAME_STOP))))
    df = df.with_columns(core.alias("core_tok"))
    # alias part: cleaned the same simple way (Latin only is enough for aliases)
    df = df.with_columns(
        pl.col("name_tok").list.join(" ").alias("name_str"),
        pl.col("core_tok").list.join(" ").alias("core_str"),
        pl.col("core_tok").list.join("").alias("concat"),
        pl.col("alias_raw").str.replace_all(r"['’`\.]", "").str.replace_all(TOKEN_SPLIT, " ")
          .str.strip_chars().alias("alias_str"),
    ).drop("alias_raw")
    return df


def normalize_addresses(df: pl.DataFrame, tdict: dict, state_native: dict | None = None) -> pl.DataFrame:
    s = (_base_clean("addr")
         .str.replace_all(r"<null>|\bnull\b|\bn/a\b|\bnone\b", " ")
         .str.replace_all(r"(\d+)\s*(?:st|nd|rd|th)\b", "$1"))
    df = df.with_columns(s.alias("_a"))
    # --- state detection per comma component (country specific maps; unknown countries -> none)
    comps = (df.select("rid", "country", pl.col("_a").str.split(",").alias("c"))
               .explode("c").with_columns(
                   pl.col("c").str.replace_all(TOKEN_SPLIT, " ").str.replace_all(r"\s+", " ")
                   .str.strip_chars()))
    parts = []
    for ctry, g in comps.group_by("country"):
        m = _state_map(ctry[0])
        if state_native:
            m = {**m, **state_native}
        parts.append(g.with_columns(pl.col("c").replace_strict(m, default=None).alias("st")) if m
                     else g.with_columns(pl.lit(None, pl.Utf8).alias("st")))
    comps = pl.concat(parts)
    agg = comps.group_by("rid").agg(
        pl.col("st").drop_nulls().last().fill_null("").alias("state"),
        pl.col("c").filter(pl.col("st").is_null() & (pl.col("c") != "")).alias("_comps"))
    df = df.join(agg, on="rid", how="left").with_columns(pl.col("state").fill_null(""))
    df = df.with_columns(pl.col("_comps").fill_null(pl.lit([], pl.List(pl.Utf8))).list.join(" ").alias("_a2"))
    df = df.with_columns(
        pl.col("_a2").str.extract_all(r"\d+").list.eval(
            pl.element().str.replace(r"^0+(\d)", "$1")).alias("addr_num"),
        pl.col("_a2").str.replace_all(r"(\d)([a-z])", "$1 $2").str.replace_all(r"([a-z])(\d)", "$1 $2")
          .str.split(" ").list.eval(pl.element().filter(pl.element() != "")).alias("addr_all"),
    )
    df = translit_tokens(df, "addr_all", tdict)
    df = _map_list(df, "addr_all", ADDR_CANON)
    df = df.with_columns(
        pl.col("addr_all").list.eval(pl.element().str.replace(r"^0+(\d)", "$1")).list.join(" ").alias("addr_str"),
        pl.col("addr_all").list.eval(
            pl.element().filter(~pl.element().str.contains(r"^\d+$") & ~pl.element().is_in(list(ADDR_STOP)))
        ).alias("addr_tok"),
        (pl.col("_a2").str.strip_chars() == "").alias("addr_missing"),
    ).drop("_a", "_comps", "_a2", "addr_all")
    return df


def normalize(df: pl.DataFrame, dicts: dict) -> pl.DataFrame:
    """dicts = {"tok": native->latin token map, "state": native state component -> code}."""
    df = normalize_names(df, dicts.get("tok", {}))
    df = normalize_addresses(df, dicts.get("tok", {}), dicts.get("state", {}))
    return df
