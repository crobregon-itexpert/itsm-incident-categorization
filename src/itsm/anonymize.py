"""Anonymization of ticket titles so the dataset can be shared with the code.

Applied to every text column that leaves the organization:
  1. URLs and e-mail addresses removed.
  2. Person, requester and store names (tokens from Solicitante, Writer, Asignado a Técnico,
     minus IT vocabulary) replaced by XXX.
  3. Digit runs replaced by 0 (store codes, cash registers, dates, phone numbers), and the
     store-name slot that follows a store code is masked:
     "ImpPixel BB24-JockeyPlaza-Caja1" -> "ImpPixel BB0-XXX-Caja0".
  4. k-anonymity on vocabulary: any word that occurs in fewer than k tickets is replaced by XXX
     (removes names and identifiers not covered by step 2, e.g. people mentioned in titles).
"""
from __future__ import annotations

import re
from collections import Counter

from .text import strip_accents

_URL = re.compile(r"(https?://|www\.)\S+")
_EMAIL = re.compile(r"\b[\w.-]+@[\w.-]+\.\w+\b")
_WORD = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+")
_DIGITS = re.compile(r"\d+")
_XXX_RUN = re.compile(r"XXX(?:[\s\-_/.,]*XXX)+")
# Store code (brand prefix + number, digits already 0) followed by a store-name segment and a hyphen
_STORE_SLOT = re.compile(r"\b([A-Z]{2,3}0)([\s\-]+)([^\-|\n]{1,40}?)(\s*-)")


def _key(tok: str) -> str:
    return strip_accents(tok.lower())


def mask_raw(text: str, names: set[str]) -> str:
    """Steps 1-3 on the original title (keeps case, accents, punctuation for BERT)."""
    if not isinstance(text, str):
        return ""
    t = _EMAIL.sub(" ", _URL.sub(" ", text))
    t = _WORD.sub(lambda m: "XXX" if _key(m.group()) in names else m.group(), t)
    t = _DIGITS.sub("0", t)
    t = _STORE_SLOT.sub(lambda m: m.group(1) + m.group(2) + "XXX" + m.group(4), t)
    return re.sub(r"\s+", " ", t).strip()


def doc_freq(texts) -> Counter:
    df = Counter()
    for t in texts:
        df.update({_key(w) for w in _WORD.findall(t)})
    return df


def mask_rare_raw(text: str, df: Counter, k: int) -> str:
    t = _WORD.sub(lambda m: m.group() if (m.group() == "XXX" or df[_key(m.group())] >= k) else "XXX", text)
    return _XXX_RUN.sub("XXX", t)


def mask_rare_clean(text: str, df: Counter, k: int) -> str:
    """Cleaned (token) text: rare tokens are dropped rather than masked."""
    return " ".join(w for w in text.split() if df[w] >= k)
