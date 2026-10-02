"""Delta between what the absorber already emitted and a longer/rewritten transcript,
decided per token (README 8a §5.3-7): a real STT capitalises and punctuates the final, so a
character prefix test would call every final a rewrite."""

import pytest

from api.services.pipecat.turns.text_delta import normalize_token, token_delta


@pytest.mark.parametrize(
    "emitted,text,expected",
    [
        ("quiero reservar para el", "quiero reservar para el sábado", "sábado"),
        # Real STT: capitalised, punctuated final over a lowercase interim → still an extension.
        ("quiero reservar para el", "Quiero reservar para el sábado.", "sábado."),
        ("hola quiero", "Hola, quiero reservar", "reservar"),
        ("quiero reservar para el sábado", "Quiero reservar para el sábado.", ""),
        ("quiero reservar", "quiero reservar", ""),
        # Genuine rewrite: a word changed.
        ("quiero reservar para el", "quiero cancelar para el sábado", None),
        # Shorter final than what was emitted: rewrite (cannot be an extension).
        ("quiero reservar para el", "quiero reservar", None),
        # Mid-word extension is not a token extension: "reservar" vs "reservado".
        ("quiero reservar", "quiero reservado hoy", None),
        ("", "hola", "hola"),
    ],
)
def test_token_delta(emitted, text, expected):
    assert token_delta(emitted, text) == expected


def test_normalize_token_strips_edge_punctuation_and_case():
    assert normalize_token("Sábado.") == "sábado"
    assert normalize_token("¿Hola,") == "hola"
    assert normalize_token("---") == ""
