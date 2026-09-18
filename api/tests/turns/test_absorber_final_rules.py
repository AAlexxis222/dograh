"""The four branches of the STT's final (spec §6.1.1 ratified form, D-11/D-12/D-13)."""

import pytest
from pipecat.frames.frames import (
    TranscriptionFrame,
    UninterruptibleFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)

from api.services.pipecat.turns.frames import HeldTranscriptionFrame
from api.tests.turns.test_absorber_unit import finals, itf, run_steps, tf


def test_a_released_held_final_is_a_transcription_that_survives_an_interruption():
    # The turn start that releases it makes the aggregator broadcast an interruption,
    # which flushes every queued interruptible frame.
    assert issubclass(HeldTranscriptionFrame, UninterruptibleFrame)
    # ...and the aggregator must still treat it as an ordinary transcription.
    assert issubclass(HeldTranscriptionFrame, TranscriptionFrame)


@pytest.mark.asyncio
async def test_rewrite_with_local_turn_open_replaces_instead_of_appending():
    # D-11: the aggregator would concatenate the promoted partial and the rewritten final.
    absorber, rec = await run_steps(
        [
            ("down", UserStartedSpeakingFrame()),
            ("up", UserStartedSpeakingFrame()),
            ("down", itf("quiero reservar para el")),
            ("up", VADUserStoppedSpeakingFrame(stop_secs=0.2)),
            ("down", tf("Quiero cancelar para el sábado.")),  # a word changed → rewrite
            ("down", UserStoppedSpeakingFrame()),
        ]
    )
    assert finals(rec) == [
        ("TranscriptionFrame", "quiero reservar para el", False, "flux"),
        ("TranscriptionReplaceFrame", "Quiero cancelar para el sábado.", None, "flux"),
    ]
    assert absorber.stats["rewrite_replaced"] == 1


@pytest.mark.asyncio
async def test_orphan_extension_is_emitted_as_new_message():
    # D-12 (S5/S8): the local turn closed before the final; the tail is real speech.
    absorber, rec = await run_steps(
        [
            ("down", UserStartedSpeakingFrame()),
            ("up", UserStartedSpeakingFrame()),
            ("down", itf("quiero reservar para el")),
            ("up", VADUserStoppedSpeakingFrame(stop_secs=0.2)),
            ("up", UserStoppedSpeakingFrame()),  # local turn closed
            ("down", tf("Quiero reservar para el sábado.")),
            ("down", UserStoppedSpeakingFrame()),
        ]
    )
    assert [d[1] for d in finals(rec)] == ["quiero reservar para el", "sábado."]
    assert finals(rec)[1][2] is True
    assert absorber.stats["orphan_emitted"] == 1
    assert absorber.stats["dup_avoided"] == 0  # the tail was delivered, not swallowed


@pytest.mark.asyncio
async def test_orphan_rewrite_is_emitted_whole_as_new_message():
    absorber, rec = await run_steps(
        [
            ("down", UserStartedSpeakingFrame()),
            ("up", UserStartedSpeakingFrame()),
            ("down", itf("quiero reservar para el")),
            ("up", VADUserStoppedSpeakingFrame(stop_secs=0.2)),
            ("up", UserStoppedSpeakingFrame()),
            ("down", tf("Quiero cancelar para el sábado.")),
            ("down", UserStoppedSpeakingFrame()),
        ]
    )
    assert [d[1] for d in finals(rec)] == [
        "quiero reservar para el",
        "Quiero cancelar para el sábado.",
    ]
    assert absorber.stats["orphan_rewrite_emitted"] == 1


@pytest.mark.asyncio
async def test_orphan_dup_is_dropped_silently():
    absorber, rec = await run_steps(
        [
            ("down", UserStartedSpeakingFrame()),
            ("up", UserStartedSpeakingFrame()),
            ("down", itf("quiero reservar")),
            ("up", VADUserStoppedSpeakingFrame(stop_secs=0.2)),
            ("up", UserStoppedSpeakingFrame()),
            ("down", tf("Quiero reservar.")),
            ("down", UserStoppedSpeakingFrame()),
        ]
    )
    assert len(finals(rec)) == 1
    assert absorber.stats["dup_avoided"] == 1


@pytest.mark.asyncio
async def test_final_with_nothing_emitted_and_no_local_turn_is_held_then_released_as_message():
    # D-13 (Sghost): no interim, no local turn → not passed through; after hold_ms it is
    # delivered on its own so real speech is never lost when the local VAD missed it.
    absorber, rec = await run_steps(
        [
            ("down", UserStartedSpeakingFrame()),
            ("down", tf("hola")),
            ("down", UserStoppedSpeakingFrame()),
            (0.1, None),
        ],
        hold_ms=200,
    )
    # run_steps sleeps 0.3 s before EndFrame → the hold (0.2 s) expires inside the run.
    assert finals(rec) == [("HeldTranscriptionFrame", "hola", True, "flux")]
    assert absorber.stats["final_held_no_local_turn"] == 1
    assert absorber.stats["held_released_as_message"] == 1


@pytest.mark.asyncio
async def test_held_final_is_released_into_the_next_local_turn():
    absorber, rec = await run_steps(
        [
            ("down", UserStartedSpeakingFrame()),
            ("down", tf("hola")),
            ("down", UserStoppedSpeakingFrame()),
            ("up", UserStartedSpeakingFrame()),  # local turn opens within hold_ms
        ],
        hold_ms=1500,
    )
    assert finals(rec) == [("HeldTranscriptionFrame", "hola", True, "flux")]
    assert absorber.stats["held_released_into_turn"] == 1
    assert absorber.stats["held_released_as_message"] == 0


@pytest.mark.asyncio
async def test_a_held_final_is_not_lost_when_the_next_flux_turn_is_held_too():
    # Flux's finals are cumulative only inside one turn: turn B's final does not carry
    # turn A's utterance, so holding B must release A instead of dropping it.
    absorber, rec = await run_steps(
        [
            ("down", UserStartedSpeakingFrame()),
            ("down", tf("hola")),
            ("down", UserStoppedSpeakingFrame()),
            ("down", UserStartedSpeakingFrame()),  # Flux turn B, still no local turn
            ("down", tf("qué tal")),
            ("down", UserStoppedSpeakingFrame()),
        ],
        hold_ms=1500,
    )
    assert [d[1] for d in finals(rec)] == ["hola", "qué tal"]
    assert absorber.stats["final_held_no_local_turn"] == 2
    assert absorber.stats["held_released_as_message"] == 2


@pytest.mark.asyncio
async def test_hold_zero_drops_nothing_but_delivers_immediately_as_message():
    absorber, rec = await run_steps(
        [
            ("down", UserStartedSpeakingFrame()),
            ("down", tf("hola")),
            ("down", UserStoppedSpeakingFrame()),
        ],
        hold_ms=0,
    )
    assert finals(rec) == [("HeldTranscriptionFrame", "hola", True, "flux")]
    assert absorber.stats["held_released_as_message"] == 1


@pytest.mark.asyncio
async def test_end_frame_flushes_a_held_final():
    absorber, rec = await run_steps(
        [
            ("down", UserStartedSpeakingFrame()),
            ("down", tf("hola")),
            ("down", UserStoppedSpeakingFrame()),
        ],
        hold_ms=10000,
    )
    # EndFrame arrives 0.3 s later, long before the hold expires: nothing may be lost.
    assert finals(rec) == [("HeldTranscriptionFrame", "hola", True, "flux")]
    assert absorber.stats["held_released_as_message"] == 1


@pytest.mark.asyncio
async def test_second_final_for_same_flux_turn_is_treated_as_extension_not_new_turn():
    # §5.3-9: not observed with Flux; if it happens, extend the closed turn instead of
    # opening a new one with the whole text.
    absorber, rec = await run_steps(
        [
            ("down", UserStartedSpeakingFrame()),
            ("up", UserStartedSpeakingFrame()),
            ("down", itf("hola quiero")),
            ("up", VADUserStoppedSpeakingFrame(stop_secs=0.2)),
            ("down", tf("hola quiero reservar")),
            (
                "down",
                tf("hola quiero reservar hoy"),
            ),  # second final, no new StartOfTurn
            ("down", UserStoppedSpeakingFrame()),
        ]
    )
    assert [d[1] for d in finals(rec)] == ["hola quiero", "reservar", "hoy"]
    assert absorber.stats["second_final_retained"] == 1
