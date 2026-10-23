import json
import logging
import re

import pytest
from loguru import logger

from api import logging_config
from api.errors import failure
from api.errors.failure import redact_credentials, redact_failure_message
from api.services.security.redaction import known_secret_values, redact

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
    [
        "+34 612 345 678",
        "+34612345678",
        "%2B34612345678",
        "0034612345678",
        "(+34) 612 345 678",
        "+1 (415) 555-2671",
        "612345678",
        "612 34 56 78",
        "612 345 678",
        "612-345-678",
        "612.34.56.78",
        "34612345678",
        "+491234567890123",
        "00491234567890123",
        "612.345.678",
    ],
)
def test_masks_spanish_and_e164_phones(phone):
    assert redact(f"llamando a {phone} ahora") == "llamando a <redacted:phone> ahora"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("msg=call+me+at+612345678", "msg=call+me+at+<redacted:phone>"),
        ("a=%22%2B34612345678", "a=%22<redacted:phone>"),
        ("Hello%20%2B34612345678", "Hello%20<redacted:phone>"),
        ("x%3A%2b34612345678", "x%3A<redacted:phone>"),
        ("q=%252B34612345678", "q=<redacted:phone>"),
        ("from sip:34612345678@host.example", "from sip:<redacted:phone>@host.example"),
        ("to tel:+34612345678 now", "to tel:<redacted:phone> now"),
        ("to TEL:%2B34612345678 now", "to TEL:<redacted:phone> now"),
        ("from sips:1001@pbx", "from sips:<redacted:phone>@pbx"),
    ],
)
def test_masks_digits_of_sip_and_tel_uris(text, expected):
    assert redact(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "at 2026-10-07 12:34:56 done",
        "at 2026-10-07T12:34:56Z done",
        "at 2026-10-07T12:34:56.789+02:00 done",
        "run 123e4567-e89b-12d3-a456-426614174000 done",
        "order 1234567890123 done",
        "latency 0.712345678 ms",
        "ts 1759856123.612345678 done",
        "+1234567890123456",
        "took 712.345678 s",
        "latency=812.123456",
        "amount 71234.5678",
        "v=6543.21987",
    ],
)
def test_keeps_dates_times_and_ids(text):
    assert redact(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "authorızation=abcdef123",
        "sıgnature=abcdef123",
        "sİgnature=abcdef123",
        "x-api-kEy: abcdef123",
        "no secret name here",
    ],
)
def test_marker_gate_matches_ungated_redaction(text, monkeypatch):
    gated = redact_failure_message(text)
    monkeypatch.setattr(failure, "_SECRET_NAME_MARKERS", ("",))  # gate always open
    assert gated == redact_failure_message(text) == redact_credentials(text)
    if "=" in text or ":" in text:
        assert "abcdef123" not in gated


def test_masks_registered_secret_values():
    out = redact("value canary-secret-xyz here", secrets=("canary-secret-xyz",))
    assert out == "value <redacted:secret> here"


def test_known_secret_values_are_longest_first(monkeypatch):
    # A secret that is a prefix of another must not leave the tail in clear.
    monkeypatch.setattr(
        "api.services.security.redaction.active_secret_values",
        lambda: ["abcdefgh", "abcdefghTAILTAIL"],
    )
    values = known_secret_values()
    assert values == ("abcdefghTAILTAIL", "abcdefgh")
    assert "TAIL" not in redact("x abcdefghTAILTAIL y", secrets=values)


@pytest.mark.parametrize(
    ("text", "leaked"),
    [
        ("Authorization: Basic dXNlcjpwYXNzd29yZA==", "dXNlcjpwYXNz"),
        ("authorization: bearer abc.DEF_123/xyz+q==", "abc.DEF_123"),
        ("GET /v1?api_key=hunter2hunter2&x=1", "hunter2hunter2"),
        ("GET /v1?foo=1&access_token=hunter2hunter2", "hunter2hunter2"),
        ('cfg password="hunter2 hunter2" end', "hunter2"),
        ("cfg client_secret=hunter2hunter2 end", "hunter2hunter2"),
        ("key sk_live_ABCDEF1234567890 end", "ABCDEF1234567890"),
        ("db postgresql+asyncpg://user:hunter2@db:5432/x", "hunter2"),
        ("cache redis://:hunter2@cache:6379/0", "hunter2"),
        ("mq amqp://guest:hunter2@mq/", "hunter2"),
    ],
)
def test_masks_credentials_in_any_shape(text, leaked):
    out = redact(text, secrets=())
    assert leaked not in out
    assert out.split()[0] == text.split()[0]  # surrounding text survives


def test_keeps_urls_without_userinfo():
    text = "GET https://example.com/path@foo and redis://cache:6379/0"
    assert redact(text, secrets=()) == text


def test_active_secret_values_reads_secret_named_env(monkeypatch):
    from api.services.security.redaction import active_secret_values

    secret_named = {
        "CANARY_API_KEY": "canary-api-key-value",
        "AWS_SECRET_ACCESS_KEY": "canary-aws-secret-value",
        "STACK_SECRET_SERVER_KEY": "canary-stack-secret-value",
        "OSS_JWT_SECRET": "canary-jwt-secret-value",
        "DB_PASSWORD": "canary-db-password-value",
        "TWILIO_AUTH_TOKEN": "canary-twilio-token-value",
        "canary_lower_token": "canary-lower-token-value",
    }
    for name, value in secret_named.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("CANARY_TOKEN", "short")
    monkeypatch.setenv("CANARY_FLAG", "canary-not-secret")
    monkeypatch.setenv("KEYBOARD_LAYOUT", "canary-not-secret-either")
    values = active_secret_values()
    assert set(secret_named.values()) <= set(values)
    assert "short" not in values
    assert "canary-not-secret" not in values
    assert "canary-not-secret-either" not in values


def test_credential_box_tokens_are_redacted(monkeypatch):
    import base64
    import os

    from api.services.security import credential_box

    monkeypatch.setenv(
        "CREDENTIALS_MASTER_KEY", "k1=" + base64.b64encode(os.urandom(32)).decode()
    )
    credential_box.load_keys.cache_clear()
    try:
        token = credential_box.encrypt(b"canary-plaintext", aad=b"t:1:c:$")
    finally:
        credential_box.load_keys.cache_clear()
    out = redact(f"stored value={token} for row 1", secrets=())
    assert token not in out and token.split(":")[3] not in out
    assert out == "stored value=<redacted:credential_token> for row 1"


def test_each_credential_master_key_is_redacted_alone(monkeypatch):
    # Canary keys from a rotation list: the whole variable is a known secret, but a
    # single key, raw or as id=base64, must be one too.
    active, previous = (
        "QUNUSVZFLUNBTkFSWS1LRVktMzItQllURVMtLS0tLS0=",
        "UFJFVklPVVMtQ0FOQVJZLUtFWS0zMi1CWVRFUy0tLS0=",
    )
    monkeypatch.setenv("CREDENTIALS_MASTER_KEY", f"k2={active}, k1={previous}")
    out = redact(f"a={active} b=k1={previous}")
    assert active not in out and previous not in out
    assert "k1=" not in out


@pytest.mark.parametrize(
    ("key_id", "glue"),
    [
        ("k1", "x"),
        ("k1", "_"),
        ("k1", "1"),
        ("k1", "key"),
        ("k1", "/"),
        ("k1", "+"),
        # A key id ending in "v1": one whole match, never a second one starting inside the token.
        ("kv1", ""),
        ("kv1", "x"),
        # A word ending in "v1" right before the token must not start the match early (it would leave the
        # ciphertext in clear), nor must a "v1:" look-alike a few parts before it.
        ("k1", "workflow_v1:"),
        ("k1", "dev1:"),
        ("k1", "kv1:"),
        ("k1", "a v1:x:y:"),
    ],
)
def test_credential_box_tokens_are_redacted_whatever_precedes_them(
    monkeypatch, key_id, glue
):
    import base64
    import os

    from api.services.security import credential_box

    monkeypatch.setenv(
        "CREDENTIALS_MASTER_KEY",
        f"{key_id}=" + base64.b64encode(os.urandom(32)).decode(),
    )
    credential_box.load_keys.cache_clear()
    try:
        token = credential_box.encrypt(b"canary-plaintext", aad=b"t:1:c:$")
    finally:
        credential_box.load_keys.cache_clear()
    out = redact(f"stored {glue}{token} end", secrets=())
    assert token.split(":")[3] not in out
    assert out == f"stored {glue}<redacted:credential_token> end"


@pytest.mark.parametrize(
    "text",
    [
        "dev1:host:12:34",
        "eleven_v1:a:b:c",
        # Only one of the two shape rules fails: a short third part, then a short fourth part.
        "build_v1:prod:eu:0123456789abcdefghijklmnop",
        "release_v1:eu:0123456789abcdef:42",
    ],
)
def test_v1_look_alikes_that_are_not_tokens_stay(text):
    assert redact(f"at {text} now", secrets=()) == f"at {text} now"


def test_derived_forms_of_a_credential_master_key_are_redacted(monkeypatch):
    import base64
    import os

    raw = os.urandom(32)
    # The URL-safe form differs from the standard one only where "+" or "/" appear.
    while not {"+", "/"} <= set(base64.b64encode(raw).decode()):
        raw = os.urandom(32)
    standard = base64.b64encode(raw).decode()
    monkeypatch.setenv("CREDENTIALS_MASTER_KEY", f"k1={standard}")
    secrets = known_secret_values()
    forms = {
        "unpadded": standard.rstrip("="),
        "urlsafe": base64.urlsafe_b64encode(raw).decode(),
        "urlsafe-unpadded": base64.urlsafe_b64encode(raw).decode().rstrip("="),
        "hex": raw.hex(),
        "hex-upper": raw.hex().upper(),
    }
    for name, form in forms.items():
        out = redact(f"leaked {form} end", secrets)
        assert form not in out, name
        assert out == "leaked <redacted:secret> end", (name, out)


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


def test_secret_set_is_computed_once_at_setup(capsys, monkeypatch, real_logging):
    calls = []

    def counting_active_secret_values():
        calls.append(1)
        return {"canary-secret-xyz"}

    monkeypatch.setattr(
        "api.services.security.redaction.active_secret_values",
        counting_active_secret_values,
    )
    monkeypatch.setattr(logging_config, "SERIALIZE_LOG_OUTPUT", False)
    logging_config.setup_logging()
    for i in range(5):
        logger.info("record {} carries canary-secret-xyz", i)
    logger.complete()

    out = capsys.readouterr().out
    assert "canary-secret-xyz" not in out
    assert out.count("<redacted:secret>") == 5
    assert len(calls) == 1


def _token(monkeypatch, key_id="k1", plaintext=b"canary-plaintext"):
    import base64
    import os

    from api.services.security import credential_box

    monkeypatch.setenv(
        "CREDENTIALS_MASTER_KEY",
        f"{key_id}=" + base64.b64encode(os.urandom(32)).decode(),
    )
    credential_box.load_keys.cache_clear()
    try:
        return credential_box.encrypt(plaintext, aad=b"t:1:c:[]")
    finally:
        credential_box.load_keys.cache_clear()


def _token_parts_left(token, out):
    return [part for part in token.split(":")[1:] if part in out]


def test_a_token_in_an_unquoted_bearer_header_is_redacted_whole(monkeypatch):
    # VOZ-N0-24c: the bearer rule stops at ":", so it must not see the token first.
    token = _token(monkeypatch)
    out = redact(f"Authorization: Bearer {token}", secrets=())
    assert not _token_parts_left(token, out), out
    assert out.endswith(" <redacted:credential_token>"), out


def test_a_token_whose_key_id_looks_like_a_provider_key_is_redacted_whole(
    monkeypatch,
):
    token = _token(monkeypatch, key_id="sk-live-rotation-2026")
    out = redact(f"value={token} end", secrets=())
    assert not _token_parts_left(token, out), out
    assert out == "value=<redacted:credential_token> end"


def test_two_tokens_glued_together_are_both_redacted(monkeypatch):
    # Plaintexts of 14 and 29 bytes (+16 tag bytes) give unpadded base64, so nothing marks where the first ends.
    first = _token(monkeypatch, plaintext=b"x" * 14)
    second = _token(monkeypatch, plaintext=b"y" * 29)
    assert not first.endswith("=") and not second.endswith("=")
    out = redact(f"a={first}{second} b", secrets=())
    assert not _token_parts_left(first, out) and not _token_parts_left(second, out), out
    assert out == "a=<redacted:credential_token><redacted:credential_token> b"
