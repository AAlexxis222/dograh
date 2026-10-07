import json
import logging
import re

import pytest
from loguru import logger

from api import logging_config
from api.services.security.redaction import redact

CANARY_KEY = "sk-live-ABCDEF1234567890"


def test_masks_provider_keys_and_bearer_tokens():
    out = redact(
        f"key={CANARY_KEY} auth=Bearer eyJhbGciOi.abc.def url=wss://u:p@host/x"
    )
    assert CANARY_KEY not in out
    assert "eyJhbGciOi" not in out
    assert "u:p@" not in out


@pytest.mark.parametrize(
    "phone",
    ["+34 612 345 678", "+34612345678", "612345678", "612 34 56 78", "612 345 678"],
)
def test_masks_spanish_and_e164_phones(phone):
    out = redact(f"llamando a {phone} ahora")
    assert "612" not in out
    assert out.startswith("llamando a ") and out.endswith(" ahora")


@pytest.mark.parametrize(
    "text",
    [
        "at 2026-10-07 12:34:56 done",
        "at 2026-10-07T12:34:56Z done",
        "at 2026-10-07T12:34:56.789+02:00 done",
        "run 123e4567-e89b-12d3-a456-426614174000 done",
        "order 1234567890123 done",
    ],
)
def test_keeps_dates_times_and_ids(text):
    assert redact(text) == text


def test_masks_registered_secret_values(monkeypatch):
    monkeypatch.setattr(
        "api.services.security.redaction.known_secret_values",
        lambda: {"canary-secret-xyz"},
    )
    assert "canary-secret-xyz" not in redact("value canary-secret-xyz here")


def test_active_secret_values_reads_secret_named_env(monkeypatch):
    from api.services.configuration.secrets_registry import active_secret_values

    monkeypatch.setenv("CANARY_API_KEY", "canary-long-value")
    monkeypatch.setenv("CANARY_TOKEN", "short")
    monkeypatch.setenv("CANARY_FLAG", "canary-not-secret")
    values = active_secret_values()
    assert "canary-long-value" in values
    assert "short" not in values
    assert "canary-not-secret" not in values


@pytest.fixture
def real_logging(monkeypatch):
    """Run the real setup_logging (it returns early under ENVIRONMENT=test) and
    put loguru and stdlib logging back as they were afterwards."""
    core = logger._core
    saved_handlers = set(core.handlers)
    saved_patcher, saved_extra = core.patcher, dict(core.extra)
    root = logging.getLogger()
    saved_root = (root.handlers[:], root.level)
    saved_named = {
        name: (lg.handlers[:], lg.level, lg.propagate)
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "mcp")
        for lg in [logging.getLogger(name)]
    }

    monkeypatch.setattr(logging_config, "ENVIRONMENT", "local")
    monkeypatch.setattr(logging_config, "LOG_FILE_PATH", None)
    monkeypatch.setattr(logging_config, "_logging_initialized", False)
    yield
    logger.complete()
    for handler_id in set(core.handlers) - saved_handlers:
        logger.remove(handler_id)
    core.patcher = saved_patcher
    core.extra.clear()
    core.extra.update(saved_extra)
    root.handlers[:], root.level = saved_root
    for name, (handlers, level, propagate) in saved_named.items():
        lg = logging.getLogger(name)
        lg.handlers[:], lg.level, lg.propagate = handlers, level, propagate


def _log_failure_and_read(capsys):
    try:
        raise RuntimeError(f"boom {CANARY_KEY}")
    except RuntimeError:
        logger.exception("failed at 2026-10-07 12:34:56 for +34612345678")
    logger.complete()
    captured = capsys.readouterr()
    return captured.out + captured.err


def test_logging_redacts_message_and_exception_keeping_prefix(
    capsys, monkeypatch, real_logging
):
    monkeypatch.setattr(logging_config, "SERIALIZE_LOG_OUTPUT", False)
    logging_config.setup_logging()

    out = _log_failure_and_read(capsys)

    assert CANARY_KEY not in out
    assert "+34612345678" not in out
    assert "RuntimeError" in out
    assert "failed at 2026-10-07 12:34:56 for <redacted:phone>" in out
    assert re.match(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3} \| ", out)
    assert "test_log_redaction.py:" in out


def test_serialized_logging_redacts_and_stays_valid_json(
    capsys, monkeypatch, real_logging
):
    monkeypatch.setattr(logging_config, "SERIALIZE_LOG_OUTPUT", True)
    logging_config.setup_logging()

    out = _log_failure_and_read(capsys)

    assert CANARY_KEY not in out
    assert "+34612345678" not in out
    record = json.loads(out.strip().splitlines()[-1])["record"]
    assert record["time"]["repr"].startswith("20")
    assert "RuntimeError" in record["message"]
    assert record["file"]["name"] == "test_log_redaction.py"


def test_stdout_sink_is_asynchronous(monkeypatch, real_logging):
    monkeypatch.setattr(logging_config, "SERIALIZE_LOG_OUTPUT", False)
    before = set(logger._core.handlers)
    logging_config.setup_logging()
    (new_id,) = set(logger._core.handlers) - before
    assert logger._core.handlers[new_id]._enqueue is True
