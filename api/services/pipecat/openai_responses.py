"""OpenAI Responses LLM with declared reasoning and deferred node transitions.

Two gaps keep pipecat's Responses service from replacing chat/completions in
a workflow (VOZ-BUG-03):

- With no ``reasoning`` configured it sends ``effort: "none"``
  (``_maybe_disable_reasoning``; fixed upstream in 49d5ebab, unreleased),
  which drops quality on gpt-6-luna and is an HTTP 400 on gpt-6.1-sol. Here
  the matrix default is declared instead.
- Dograh's chat service holds a node transition the model calls in the same
  response it speaks until that speech has played (``BotStoppedSpeakingFrame``),
  so the next node's prompt never cuts the reply. The Responses service runs
  it at once; the deferral is ported here.
"""

from collections.abc import Sequence

from loguru import logger

from api.services.pipecat.openai_efforts import row_for
from pipecat.frames.frames import BotStoppedSpeakingFrame, Frame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallFromLLM
from pipecat.services.openai.responses.llm import OpenAIResponsesHttpLLMService
from pipecat.utils.text.alnum_utils import has_alnum
from pipecat.utils.types import assert_given


class DograhOpenAIResponsesLLMService(OpenAIResponsesHttpLLMService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._response_text = ""
        self._pending_node_transition_function_calls: list[FunctionCallFromLLM] = []

    def _maybe_disable_reasoning(self, params: dict):
        """Declare the matrix default where pipecat would send ``effort: none``.

        Reached only when the settings carry no reasoning: a model that does
        not reason (nothing to declare) or a service built outside the factory.
        """
        effort = row_for(assert_given(self._settings.model)).default
        if effort is not None:
            params["reasoning"] = {"effort": effort}
            params["include"] = ["reasoning.encrypted_content"]

    async def _process_context(self, context: LLMContext):
        if self._pending_node_transition_function_calls:
            logger.warning(
                f"{self}: discarding "
                f"{len(self._pending_node_transition_function_calls)} deferred "
                "node-transition calls never released by TTS"
            )
        self._pending_node_transition_function_calls = []
        self._response_text = ""
        await super()._process_context(context)

    async def _push_llm_text(self, text: str):
        self._response_text += text
        await super()._push_llm_text(text)

    async def run_function_calls(self, function_calls: Sequence[FunctionCallFromLLM]):
        """Hold a batch with a node transition until the spoken reply ends.

        The batch is kept whole, as the chat service does: splitting it would
        drop pipecat's shared function-call group. Unspeakable text never
        produces the ``BotStoppedSpeakingFrame`` that releases it, so it does
        not count; speculative calls must stay cancellable, so they run.
        """
        if (
            not self._speculation_gate.is_speculating
            and has_alnum(self._response_text)
            and any(
                self._function_is_node_transition(fc.function_name)
                for fc in function_calls
            )
        ):
            self._pending_node_transition_function_calls = list(function_calls)
            logger.debug(
                f"{self}: Deferring {len(function_calls)} node-transition "
                "function calls until after TTS"
            )
            return
        await super().run_function_calls(function_calls)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if (
            isinstance(frame, BotStoppedSpeakingFrame)
            and self._pending_node_transition_function_calls
        ):
            function_calls = self._pending_node_transition_function_calls
            self._pending_node_transition_function_calls = []
            logger.debug(
                f"{self}: Executing {len(function_calls)} deferred node-transition "
                "function calls after TTS"
            )
            await super().run_function_calls(function_calls)
