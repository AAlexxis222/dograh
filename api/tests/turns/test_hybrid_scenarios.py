"""S1-S11 of the 8a spike replayed against the production absorber, with the production
expectations of spec §6.1.1 (ratified form): one message with the full text in the aligned
scenarios, the whole sentence in S5 (orphan tail emitted), no ghost turn in Sghost, no
duplicated context in Srewrite, no dangling tasks, and Flux's turn proposals (Pipecat 1.12
``Proposed*`` frames) never reach the aggregator. Slow (~2-3 min): run in the foreground with
a 600 s timeout."""

import pytest
from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState as E
from pipecat.processors.aggregators.llm_context import LLMContext

from api.schemas.answer_supervisor import AnswerSupervisorConfig
from api.services.pipecat.answer_classification import MachineSubtype
from api.services.pipecat.processors.answer_supervisor import AnswerSupervisor
from api.tests.turns import scenarios as S
from api.tests.turns.harness import Scenario, run_scenario

PROPOSALS = (
    "DOWN:ProposedUserStartedSpeakingFrame",
    "DOWN:ProposedUserStoppedSpeakingFrame",
)


def _proposals_reaching_aggregator(r) -> int:
    return sum(r.counts[k] for k in PROPOSALS)


def _ghost(r) -> bool:
    return len(r.messages) > r.local_turns_completed


def _lost_total(r) -> int:
    return (r.transcripts_emitted - r.transcripts_to_aggregator) + r.dropped_by_absorber


def _texts(r):
    return [m[1].strip() for m in r.messages]


@pytest.mark.asyncio
@pytest.mark.parametrize("sc", [S.S1, S.S2, S.S3, S.S4, S.S8], ids=lambda s: s.id)
async def test_aligned_scenarios_one_message_full_text(sc):
    r = await run_scenario(sc, absorber=True)
    assert _texts(r) == [S.TEXT], r.events
    assert not _ghost(r)
    assert _proposals_reaching_aggregator(r) == 0, r.counts
    assert _lost_total(r) == 0
    assert r.dangling_tasks == 0


@pytest.mark.asyncio
async def test_s5_pause_mid_sentence_delivers_the_whole_sentence():
    # D-12: the local turn closes ~190 ms before Flux's final, so the tail is nobody's. The
    # spike dropped it (PART only, `orphan_text_dropped=1`); production leaves it as a message
    # of its own, which is what the orphan branch below pins. Loss is checked by `_lost_total`.
    r = await run_scenario(S.S5, absorber=True)
    assert " ".join(_texts(r)) == S.TEXT, r.events
    assert r.absorber_stats["orphan_emitted"] == 1, r.absorber_stats
    assert _lost_total(r) == 0
    assert r.dangling_tasks == 0


@pytest.mark.asyncio
async def test_s5b_detector_fails_at_the_pause_still_delivers_everything():
    r = await run_scenario(S.S5B, absorber=True)
    assert " ".join(_texts(r)) == S.TEXT, r.events
    assert _lost_total(r) == 0


@pytest.mark.asyncio
async def test_srewrite_replaces_instead_of_duplicating():
    r = await run_scenario(S.SREWRITE, absorber=True)
    joined = " ".join(_texts(r))
    assert "reservar" not in joined, r.events  # the promoted partial was replaced
    assert S.REWRITE_FINAL in joined
    assert (
        r.absorber_stats["rewrite_replaced"]
        + r.absorber_stats["orphan_rewrite_emitted"]
        == 1
    )
    assert r.dangling_tasks == 0


@pytest.mark.asyncio
async def test_sghost_no_local_turn_is_held_then_delivered_without_ghost_flag_lying():
    r = await run_scenario(S.SGHOST, absorber=True, hold_ms=500)
    assert _texts(r) == [S.TEXT]
    assert r.absorber_stats["final_held_no_local_turn"] == 1
    assert r.absorber_stats["held_released_as_message"] == 1
    # The message exists but no local analyzer closed it: the metric must say so.
    assert _ghost(r)


@pytest.mark.asyncio
@pytest.mark.parametrize("sc", [S.S6, S.S11], ids=lambda s: s.id)
async def test_bargein_one_interruption_from_local_vad(sc):
    r = await run_scenario(sc, absorber=True)
    assert _texts(r) == [S.TEXT]
    assert r.counts["UP:InterruptionFrame"] >= 1
    assert _proposals_reaching_aggregator(r) == 0, r.counts


@pytest.mark.asyncio
async def test_s7_min_words_still_one_message():
    r = await run_scenario(S.S7, absorber=True)
    assert _texts(r) == [S.TEXT]
    assert _proposals_reaching_aggregator(r) == 0, r.counts


@pytest.mark.asyncio
async def test_s9_muted_turn_then_normal_turn():
    r = await run_scenario(S.S9B, absorber=True)
    # Turn 1 is muted (MuteUntilFirstBotComplete); turn 2 behaves like S1. One message total:
    # the muted turn leaked nothing, not even the final the absorber held and released.
    assert len(_texts(r)) == 1, r.messages
    assert _texts(r)[-1] == S.TEXT, r.events
    # Pipecat 1.12: the proposals are discarded under mute too (spec B2 VOZ-AC-B2-12). One
    # that slipped through would leave ``UserTurnController._user_speaking`` set and keep the
    # local close from finalizing (``user_turn_controller.py:321-324, :449-450``).
    assert _proposals_reaching_aggregator(r) == 0, r.counts


# V19-ter, ported to Pipecat 1.12: upstream replaced the voicemail detector with the
# ``AnswerSupervisor`` (no ``VoicemailDetector`` is built anywhere under ``api/services``). It
# sits at point B, above the absorber, and classifies the aggregator's turn message
# (``answer_supervisor.py:120-128`` binds ``on_user_turn_stopped``, ``:284`` classifies its
# content), so with the hybrid on it classifies the promoted interim instead of Flux's final.
# Short on purpose: one local turn is all the classifier needs.
SVOICEMAIL = Scenario(
    id="SVOICEMAIL",
    flux=[(0, "start", ""), (800, "update", S.TEXT), (1400, "end", S.TEXT)],
    speaking=[(0, 1000)],
    verdicts=[E.COMPLETE],
    end_at=3000,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "absorber", [True, False], ids=["promoted-interim", "flux-final"]
)
async def test_answer_supervisor_classifies_either_way(absorber):
    classified: list[str] = []

    async def classify(text: str) -> MachineSubtype:
        classified.append(text)
        return MachineSubtype.CONVERSATION

    # ``human_utterance_max_ms=1``: no turn is short enough to skip the classifier.
    supervisor = AnswerSupervisor(
        AnswerSupervisorConfig(human_utterance_max_ms=1),
        context=LLMContext(),
        classify=classify,
    )
    r = await run_scenario(SVOICEMAIL, absorber=absorber, answer_supervisor=supervisor)
    # The classifier runs once either way, on the whole sentence: a promoted interim is
    # classified exactly like Flux's final.
    assert classified == [S.TEXT], (classified, r.events)
    if absorber:
        # The promoted interim was the only user text the supervisor saw: Flux's final
        # repeated it and the absorber dropped it as a duplicate.
        assert r.absorber_stats["promoted"] == 1, r.absorber_stats
        assert r.absorber_stats["dup_avoided"] == 1, r.absorber_stats


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sc", [S.S1, S.S5, S.S6, S.S7, S.S8, S.S11], ids=lambda s: s.id
)
async def test_mutation_without_absorber_breaks(sc):
    r = await run_scenario(sc, absorber=False)
    broken = _proposals_reaching_aggregator(r) > 0 or _texts(r) != [S.TEXT] or _ghost(r)
    assert broken, r.events
