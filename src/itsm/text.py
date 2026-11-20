"""Text normalization for Spanish service desk tickets."""
from __future__ import annotations

import re
import unicodedata

# Spanish stop words (NLTK list, accents stripped) so no download is needed.
SPANISH_STOPWORDS = set("""
de la que el en y a los del se las por un para con no una su al lo como mas pero sus le ya o este si porque esta
entre cuando muy sin sobre tambien me hasta hay donde quien desde todo nos durante todos uno les ni contra otros ese
eso ante ellos e esto mi antes algunos que unos yo otro otras otra el tanto esa estos mucho quienes nada muchos cual
poco ella estar estas algunas algo nosotros mi mis tu te ti tu tus ellas nosotras vosotros vosotras os mio mia mios
mias tuyo tuya tuyos tuyas suyo suya suyos suyas nuestro nuestra nuestros nuestras vuestro vuestra vuestros vuestras
esos esas estoy estas esta estamos estais estan este estes estemos esteis esten estare estaras estara estaremos
estareis estaran estaria estarias estariamos estariais estarian estaba estabas estabamos estabais estaban estuve
estuviste estuvo estuvimos estuvisteis estuvieron he has ha hemos habeis han haya hayas hayamos hayais hayan habre
habras habra habremos habreis habran habria habrias habriamos habriais habrian habia habias habiamos habiais habian
hube hubiste hubo hubimos hubisteis hubieron soy eres es somos sois son sea seas seamos seais sean sere seras sera
seremos sereis seran seria serias seriamos seriais serian era eras eramos erais eran fui fuiste fue fuimos fuisteis
fueron tengo tienes tiene tenemos teneis tienen tenga tengas tengamos tengais tengan tendre tendras tendra tendremos
tendreis tendran tendria tendrias tendriamos tendriais tendrian tenia tenias teniamos teniais tenian tuve tuviste
tuvo tuvimos tuvisteis tuvieron tened siendo sido tenido tenida tenidos tenidas
""".split())

_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_URL = re.compile(r"(https?://|www\.)\S+")
_RN_GLUED = re.compile(r"\\?r\\?n(?=[A-ZÁÉÍÓÚÑ¿¡\-\s\d:.]|$)|(?<=[a-záéíóúñ.,:;)])rn(?=[a-záéíóúñ])")


def plural_map(texts, min_count: int = 3) -> dict[str, str]:
    """Map Spanish plural tokens to their singular when the singular is in the corpus.

    boletas -> boleta, impresoras -> impresora, conexiones -> conexion.
    Data-driven (no stemmer dependency); only folds when both forms occur.
    """
    from collections import Counter
    cnt = Counter(t for s in texts for t in s.split())
    vocab = {w for w, c in cnt.items() if c >= min_count}
    m = {}
    for w in cnt:
        if len(w) < 4 or not w.endswith("s"):
            continue
        if w[:-1] in vocab:                       # partes -> parte, boletas -> boleta
            m[w] = w[:-1]
        elif w.endswith("es") and len(w) > 5 and w[:-2] in vocab:   # conexiones -> conexion
            m[w] = w[:-2]
    return m


def apply_map(text: str, m: dict[str, str]) -> str:
    return " ".join(m.get(t, t) for t in text.split())


def strip_solution_template(text: str, tcfg: dict | None) -> str:
    """Remove form-field labels and the trailing contact block from a solution note."""
    if not isinstance(text, str) or not tcfg:
        return text if isinstance(text, str) else ""
    t = text
    for marker in tcfg.get("drop_from", []):
        i = t.lower().find(marker.lower())
        if i >= 0:
            t = t[:i]
    for lab in tcfg.get("labels", []):
        t = re.sub(rf"{re.escape(lab)}\s*:?", " ", t, flags=re.IGNORECASE)
    for ph in tcfg.get("phrases", []):
        t = re.sub(rf"\b{re.escape(ph)}\w*", " ", t, flags=re.IGNORECASE)
    return t


def name_stopwords(df, cols, keep) -> set[str]:
    """Tokens of people/store names found in the given columns, minus domain words."""
    keep = {strip_accents(k.lower()) for k in keep or []}
    toks = set()
    for c in cols or []:
        if c not in df.columns:
            continue
        for v in df[c].dropna().astype(str).unique():
            for t in re.sub(r"[^a-z\s]", " ", strip_accents(v.lower())).split():
                if len(t) >= 3 and t not in keep:
                    toks.add(t)
    return toks
_SPACES = re.compile(r"\s+")


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


class TextNormalizer:
    """Configurable cleaner. Same instance is used for clustering and classification."""

    def __init__(self, cfg: dict, extra_stopwords: set[str] | None = None):
        self.cfg = cfg
        self.extra_stop = extra_stopwords or set()
        self.abbrev = {k.lower(): v for k, v in (cfg.get("abbreviations") or {}).items()}
        phrases = sorted(cfg.get("boilerplate_phrases") or [], key=len, reverse=True)
        self._phrases = [re.compile(rf"\b{re.escape(strip_accents(p.lower()))}\b") for p in phrases]

    def __call__(self, text) -> str:
        if not isinstance(text, str):
            return ""
        t = _URL.sub(" ", text)
        # Exports lost the backslashes of \r\n, leaving "rn" glued to words: "rnSe valida"
        t = _RN_GLUED.sub(" ", t)
        if self.cfg.get("lowercase", True):
            t = t.lower()
        if self.cfg.get("strip_accents", True):
            t = strip_accents(t)
        t = _NON_ALNUM.sub(" ", t)
        for p in self._phrases:
            t = p.sub(" ", t)
        out = []
        min_len = self.cfg.get("min_token_len", 2)
        for tok in t.split():
            tok = self.abbrev.get(tok, tok)
            if self.cfg.get("drop_tokens_with_digits", True) and any(ch.isdigit() for ch in tok):
                continue
            if len(tok) < min_len:
                continue
            if tok == "rn" or tok in self.extra_stop:
                continue
            if self.cfg.get("remove_stopwords", True) and tok in SPANISH_STOPWORDS:
                continue
            out.append(tok)
        return _SPACES.sub(" ", " ".join(out)).strip()
