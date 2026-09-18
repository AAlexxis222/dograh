"""Token-level comparison between an already-emitted transcript and a newer one.

The hybrid absorber promotes an interim and later receives the STT's final. Whether the
final *extends* the interim (emit only the tail), *repeats* it (emit nothing) or *rewrites*
it (replace) is decided on normalised tokens: the final of a real STT is capitalised and
punctuated while the interim is not, and a character-level prefix test would call every
such final a rewrite (spike 8a, README §5.3-7).
"""

import unicodedata

_EDGE_PUNCTUATION = "".join(
    chr(c) for c in range(0x10000) if unicodedata.category(chr(c)).startswith("P")
)


def normalize_token(token: str) -> str:
    return token.strip(_EDGE_PUNCTUATION).casefold()


def _tokens(text: str) -> list[str]:
    return [t for t in text.split() if normalize_token(t)]


def token_delta(emitted: str, text: str) -> str | None:
    """What ``text`` adds beyond ``emitted``.

    Returns ``""`` when it adds nothing, ``None`` when ``text`` does not extend ``emitted``
    (a rewrite), otherwise the tail of ``text`` in its original form.
    """
    old, new = _tokens(emitted), _tokens(text)
    if len(new) < len(old):
        return None
    for a, b in zip(old, new):
        if normalize_token(a) != normalize_token(b):
            return None
    return " ".join(new[len(old) :])
