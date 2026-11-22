"""The provider-neutral refusal of a tuning knob (B0-28 shape).

Any provider's matrix or spec row raises ``KnobRefusal``; ``HTTP_STATUS`` is
the one place that maps its code to the run-start status, and
``http_refusals`` applies that mapping to a builder.
"""

from collections.abc import Callable
from functools import wraps
from typing import TypeVar

from fastapi import HTTPException


class KnobRefusal(ValueError):
    """A knob the selected model, API or build cannot take (B0-28 shape).

    ``context`` carries the extra fields a code needs (``gate`` and
    ``source`` for ``build_gate``, VOZ-AC-B1-18).
    """

    def __init__(
        self, *, code: str, where: str, reason: str, hint: str, **context: str
    ):
        super().__init__(f"{where}: {reason}; {hint}")
        self.code = code
        self.where = where
        self.reason = reason
        self.hint = hint
        self.context = context

    def as_detail(self) -> dict[str, str]:
        return {
            "code": self.code,
            "where": self.where,
            "reason": self.reason,
            "hint": self.hint,
            **self.context,
        }


# Knob refusals keep the 400 the factory always used at run start; a build
# gate is the 422 VOZ-AC-B1-18 names.
HTTP_STATUS = {
    "knob_out_of_range": 400,
    "knob_not_supported": 400,
    "api_not_supported": 400,
    "build_gate": 422,
}

_T = TypeVar("_T")


def http_refusals(build: Callable[..., _T]) -> Callable[..., _T]:
    @wraps(build)
    def wrapper(*args, **kwargs) -> _T:
        try:
            return build(*args, **kwargs)
        except KnobRefusal as e:
            raise HTTPException(
                status_code=HTTP_STATUS[e.code], detail=e.as_detail()
            ) from e

    return wrapper
