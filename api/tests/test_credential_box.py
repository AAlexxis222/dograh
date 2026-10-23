"""VOZ-N0-23: the credential_box AEAD envelope and the single registry of secret surfaces."""

import base64
import os

import pytest

from api.services.security import credential_box as cb

KEY = base64.b64encode(os.urandom(32)).decode()
OTHER_KEY = base64.b64encode(os.urandom(32)).decode()
AAD = b"organization_configurations:7:value:$.byok.openai.api_key"


@pytest.fixture(autouse=True)
def master_key(monkeypatch):
    monkeypatch.setenv("CREDENTIALS_MASTER_KEY", f"k1={KEY}")
    cb.load_keys.cache_clear()
    yield
    cb.load_keys.cache_clear()


def _assert_b0_28(error, code):
    assert error.code == code
    assert error.where and error.reason and error.hint


def test_round_trip_with_aad():
    token = cb.encrypt(b"sk-secret", aad=AAD)
    assert token.startswith("v1:k1:")
    assert cb.decrypt(token, aad=AAD) == b"sk-secret"


def test_token_copied_to_another_row_does_not_decrypt():
    token = cb.encrypt(b"sk-secret", aad=b"organization_configurations:7:value:$.k")
    with pytest.raises(cb.CredentialUnreadable) as e:
        cb.decrypt(token, aad=b"organization_configurations:8:value:$.k")
    assert e.value.code == "credential_unreadable" and e.value.hint


def test_each_token_gets_a_fresh_nonce():
    first, second = cb.encrypt(b"sk-secret", aad=AAD), cb.encrypt(b"sk-secret", aad=AAD)
    assert first.split(":")[2] != second.split(":")[2]
    assert (
        len(base64.b64decode(first.split(":")[2])) == 12
    )  # 96-bit nonce (VOZ-AC-B6-15)


def test_encrypt_refuses_an_empty_aad():
    # An empty AAD would bind the token to no row: a copy would decrypt anywhere.
    with pytest.raises(ValueError, match="code=credential_aad_missing"):
        cb.encrypt(b"sk-secret", aad=b"")


def test_missing_or_short_master_key_fails_fast(monkeypatch):
    monkeypatch.setenv(
        "CREDENTIALS_MASTER_KEY", "k1=" + base64.b64encode(b"short").decode()
    )
    with pytest.raises(RuntimeError, match="code=credentials_master_key_invalid"):
        cb.load_keys()


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        ("", "CREDENTIALS_MASTER_KEY is missing"),
        (" , ", "CREDENTIALS_MASTER_KEY is missing"),
        (KEY, "is not 'key_id=base64'"),
        (f"={KEY}", "key id must be"),
        (f"k:1={KEY}", "key id must be"),
        (f"k1={KEY},k1={OTHER_KEY}", "key id appears twice"),
        # A lenient decoder would drop the "!" and accept the key; the boundary must not.
        (f"k1={KEY[:20]}!{KEY[20:]}", "key is not valid base64"),
        # A zero-width space pasted with the key: str.strip keeps it, and b64decode raises a plain ValueError.
        (f"k1={KEY}\u200b", "key is not valid base64"),
        ("k1=" + base64.b64encode(b"x" * 31).decode(), "key is 31 bytes, need 32"),
        ("k1=" + base64.b64encode(b"x" * 33).decode(), "key is 33 bytes, need 32"),
    ],
    ids=[
        "empty",
        "only-separators",
        "no-equals",
        "empty-id",
        "colon-in-id",
        "duplicate-id",
        "bad-base64",
        "non-ascii",
        "31-bytes",
        "33-bytes",
    ],
)
def test_master_key_is_validated_at_the_boundary(monkeypatch, value, reason):
    monkeypatch.setenv("CREDENTIALS_MASTER_KEY", value)
    with pytest.raises(cb.MasterKeyInvalid) as e:
        cb.load_keys()
    _assert_b0_28(e.value, "credentials_master_key_invalid")
    assert reason in e.value.reason
    assert str(e.value).startswith("code=credentials_master_key_invalid where=")
    # The key material never reaches the message, even when the operator pasted it in the wrong shape.
    for secret in (KEY, OTHER_KEY):
        assert secret not in str(e.value)


def test_the_active_key_encrypts_and_older_keys_still_decrypt(monkeypatch):
    old_token = cb.encrypt(b"sk-old", aad=AAD)
    monkeypatch.setenv("CREDENTIALS_MASTER_KEY", f"k2={OTHER_KEY}, k1={KEY}")
    cb.load_keys.cache_clear()
    assert cb.decrypt(old_token, aad=AAD) == b"sk-old"
    assert cb.encrypt(b"sk-new", aad=AAD).startswith("v1:k2:")


def test_without_a_key_encrypt_and_decrypt_fail_named_and_lazily(monkeypatch):
    token = cb.encrypt(b"sk-secret", aad=AAD)
    monkeypatch.delenv("CREDENTIALS_MASTER_KEY")
    cb.load_keys.cache_clear()
    for call in (
        lambda: cb.encrypt(b"sk-secret", aad=AAD),
        lambda: cb.decrypt(token, aad=AAD),
    ):
        with pytest.raises(
            cb.MasterKeyInvalid, match="code=credentials_master_key_invalid"
        ):
            call()


def _tampered_ciphertext(token):
    version, key_id, nonce, ct = token.split(":")
    raw = bytearray(base64.b64decode(ct))
    raw[0] ^= 1
    return ":".join((version, key_id, nonce, base64.b64encode(bytes(raw)).decode()))


def _with_part(token, index, value):
    parts = token.split(":")
    parts[index] = value
    return ":".join(parts)


@pytest.mark.parametrize(
    ("make_token", "reason"),
    [
        (lambda t: None, "not a string"),
        (lambda t: "sk-plaintext-value", "not a v1 token"),
        (lambda t: t + ":extra", "not a v1 token"),
        (lambda t: _with_part(t, 0, "v2"), "not a v1 token"),
        (lambda t: _with_part(t, 1, "k9"), "key id is not in CREDENTIALS_MASTER_KEY"),
        (lambda t: _with_part(t, 2, "@@@@"), "not valid base64"),
        (lambda t: _with_part(t, 3, "@@@@"), "not valid base64"),
        (
            lambda t: _with_part(t, 2, "\u00e9" + t.split(":")[2][1:]),
            "not valid base64",
        ),
        (lambda t: _with_part(t, 3, t.split(":")[3] + "\u00e9"), "not valid base64"),
        (
            lambda t: _with_part(t, 2, base64.b64encode(os.urandom(16)).decode()),
            "nonce is 16 bytes, need 12",
        ),
        (_tampered_ciphertext, "authentication failed"),
        (lambda t: _with_part(t, 3, ""), "authentication failed"),
    ],
    ids=[
        "none",
        "plaintext",
        "five-parts",
        "v2",
        "unknown-key-id",
        "bad-nonce-b64",
        "bad-ct-b64",
        "non-ascii-nonce",
        "non-ascii-ct",
        "long-nonce",
        "tampered",
        "empty-ct",
    ],
)
def test_every_unreadable_token_raises_credential_unreadable(make_token, reason):
    token = make_token(cb.encrypt(b"sk-secret", aad=AAD))
    with pytest.raises(cb.CredentialUnreadable) as e:
        cb.decrypt(token, aad=AAD)
    _assert_b0_28(e.value, "credential_unreadable")
    assert reason in e.value.reason
    assert e.value.where == AAD.decode()
    assert "sk-plaintext-value" not in str(e.value)


def test_aad_for_renders_the_concrete_path_as_compact_json():
    aad = cb.aad_for(
        "workflow_definitions",
        7,
        "workflow_json",
        ("nodes", 0, "data", "custom_headers", 1, "value"),
    )
    assert (
        aad
        == b'workflow_definitions:7:workflow_json:["nodes",0,"data","custom_headers",1,"value"]'
    )
    token = cb.encrypt(b"sk-secret", aad=aad)
    assert cb.decrypt(token, aad=aad) == b"sk-secret"


def test_aad_for_keeps_paths_that_collide_as_joined_strings_apart():
    # Dotted or colon-joined, ("a.b",), ("a:b",) and ("a", "b") would be the same string.
    aads = {cb.aad_for("t", 1, "c", path) for path in (("a.b",), ("a:b",), ("a", "b"))}
    assert len(aads) == 3


@pytest.mark.parametrize("path", [("nodes", "*", "value"), ("model", "**", "api_key")])
def test_aad_for_refuses_a_wildcard_path(path):
    # A token binds to one concrete leaf; a registry pattern is not a row address.
    with pytest.raises(ValueError, match="code=credential_aad_wildcard"):
        cb.aad_for("t", 1, "c", path)


def test_aad_for_refuses_a_string_path():
    # A str is a Sequence too: "$.byok.api_key" would be rendered character by character.
    with pytest.raises(ValueError, match="code=credential_aad_path_str"):
        cb.aad_for("t", 1, "c", "$.byok.api_key")


def test_the_concrete_walk_records_list_positions_under_wildcards():
    from api.services.security.secret_walk import iter_concrete_leaves

    document = {"s": {"x": [{"api_key": "a"}, {"deep": [{"api_key": "b"}]}]}}
    assert set(iter_concrete_leaves(document, ("s", "*", "api_key"))) == {
        (("s", "x", 0, "api_key"), "a")
    }
    assert set(iter_concrete_leaves(document, ("s", "**", "api_key"))) == {
        (("s", "x", 0, "api_key"), "a"),
        (("s", "x", 1, "deep", 0, "api_key"), "b"),
    }


def test_every_leaf_of_a_graph_walk_gets_its_own_aad():
    # The walk N0-24 encrypts over: each leaf's concrete path (list indexes included) binds its token to that leaf.
    from api.services.security.secret_surfaces import WORKFLOW_GRAPH_SECRET_PATHS
    from api.services.security.secret_walk import iter_concrete_leaves

    graph = {
        "nodes": [
            {
                "data": {
                    "custom_headers": [
                        {"key": "Authorization", "value": "Bearer a"},
                        {"key": "X-Api-Key", "value": "b"},
                    ]
                }
            },
            {"data": {"qa_api_key": "c"}},
            {"data": {"custom_headers": [{"key": "X-Other", "value": "d"}]}},
        ]
    }
    leaves = [
        leaf
        for pattern in WORKFLOW_GRAPH_SECRET_PATHS
        for leaf in iter_concrete_leaves(graph, pattern)
    ]
    assert len(leaves) == 4
    assert set(leaves) == {
        (("nodes", 0, "data", "custom_headers", 0, "value"), "Bearer a"),
        (("nodes", 0, "data", "custom_headers", 1, "value"), "b"),
        (("nodes", 1, "data", "qa_api_key"), "c"),
        (("nodes", 2, "data", "custom_headers", 0, "value"), "d"),
    }
    aads = [
        cb.aad_for("workflow_definitions", 7, "workflow_json", p) for p, _ in leaves
    ]
    assert len(set(aads)) == len(leaves) == 4
    token = cb.encrypt(b"Bearer a", aad=aads[0])
    for other in aads[1:]:
        with pytest.raises(cb.CredentialUnreadable):
            cb.decrypt(token, aad=other)


# --- Registry ---------------------------------------------------------------------------------------------------


def test_registry_covers_every_b0_16_surface():
    from api.services.security.secret_surfaces import SURFACES

    tables = {s.table for s in SURFACES}
    # The plan wrote "webhooks"; custom_headers lives in webhook_deliveries (api/db/models.py).
    assert {
        "organization_configurations",
        "workflow_runs",
        "workflow_definitions",
        "user_configurations",
        "telephony_configurations",
        "integrations",
        "external_credentials",
        "webhook_deliveries",
    } <= tables


def test_every_surface_names_a_real_table_and_column():
    from api.db.models import Base
    from api.services.security.secret_surfaces import SURFACES

    for surface in SURFACES:
        table = Base.metadata.tables.get(surface.table)
        assert table is not None and surface.column in table.c, surface


def test_the_org_credential_keys_are_registered_whole():
    from api.enums import OrganizationConfigurationKey as Key
    from api.services.security.secret_surfaces import SURFACES

    whole = {
        s.row_key
        for s in SURFACES
        if (s.table, s.column, s.json_path)
        == ("organization_configurations", "value", ())
    }
    assert whole == {
        Key.MODEL_CONFIGURATION_V2.value,
        Key.TELEPHONY_CONFIGURATION.value,
        Key.TWILIO_CONFIGURATION.value,
        Key.LANGFUSE_CREDENTIALS.value,
    }
    valid = {k.value for k in Key}
    assert all(s.row_key in valid for s in SURFACES if s.row_key is not None)


def _paths(table, column):
    from api.services.security.secret_surfaces import SURFACES

    return [s.json_path for s in SURFACES if (s.table, s.column) == (table, column)]


def test_secret_paths_derive_from_the_registry():
    from api.services.configuration.secrets_registry import SECRET_PATHS
    from api.services.security.secret_surfaces import WORKFLOW_CONFIGURATION_COLUMNS

    # Every column that holds a workflow configuration document carries the same secret leaves.
    assert {table for table, _ in WORKFLOW_CONFIGURATION_COLUMNS} == {
        "workflow_definitions",
        "workflows",
        "workflow_runs",
    }
    for table, column in WORKFLOW_CONFIGURATION_COLUMNS:
        assert tuple(_paths(table, column)) == SECRET_PATHS


def test_org_masking_reads_the_registry_leaf_names():
    # ai_model_configuration.py masks every SERVICE_SECRET_FIELDS leaf of the MODEL_CONFIGURATION_V2 row, at any depth.
    from api.services.configuration import masking
    from api.services.security import secret_surfaces

    assert masking.SERVICE_SECRET_FIELDS is secret_surfaces.SECRET_LEAF_NAMES
    assert () in [
        s.json_path
        for s in secret_surfaces.SURFACES
        if s.row_key == "MODEL_CONFIGURATION_V2"
    ]


WORKFLOW_GRAPH_COLUMNS = {
    ("workflow_definitions", "workflow_json"),
    ("workflows", "workflow_definition"),
    ("workflow_templates", "template_json"),
}


def test_every_node_secret_field_is_registered():
    # Not derived: integration packages declare their own sensitive_fields at load time, so this guard is the link.
    from api.services.configuration.masking import _NODE_SECRET_FIELDS
    from api.services.integrations import all_packages

    declared = {f for fields in _NODE_SECRET_FIELDS.values() for f in fields}
    declared |= {
        f
        for package in all_packages()
        for node in package.nodes
        for f in node.sensitive_fields
    }
    assert {
        "qa_api_key",
        "noveum_api_key",
        "paygent_api_key",
        "tuner_api_key",
    } <= declared
    for table, column in WORKFLOW_GRAPH_COLUMNS:
        assert {("nodes", "data", field) for field in declared} <= set(
            _paths(table, column)
        )


def test_webhook_node_custom_header_values_are_registered_in_every_graph_column():
    # The source of webhook_deliveries.custom_headers: run_integrations.py copies these values verbatim.
    from api.services.security.secret_walk import iter_concrete_leaves

    graph = {
        "nodes": [
            {
                "data": {
                    "custom_headers": [
                        {"key": "Authorization", "value": "Bearer canary"}
                    ]
                }
            },
            {"data": {"custom_headers": [{"key": "X-Api-Key", "value": "canary-2"}]}},
        ]
    }
    for table, column in WORKFLOW_GRAPH_COLUMNS:
        paths = [p for p in _paths(table, column) if "custom_headers" in p]
        assert paths, (table, column)
        # Lists are transparent in the registry walk: the path must reach every header value, never a header name.
        leaves = {v for p in paths for _, v in iter_concrete_leaves(graph, p)}
        assert leaves == {"Bearer canary", "canary-2"}, (table, column)
    # The delivery-row copy stays registered too.
    assert () in _paths("webhook_deliveries", "custom_headers")


def test_tool_static_header_values_are_registered():
    # HttpApiConfig.headers and HttpTransferResolverConfig.headers (api/schemas/tool.py): dict[str, str] sent on
    # every request.
    from api.services.security.secret_walk import iter_concrete_leaves

    definitions = [
        {"type": "http_api", "config": {"headers": {"Authorization": "Bearer canary"}}},
        {
            "type": "transfer_call",
            "config": {"resolver": {"headers": {"X-Api-Key": "canary-2"}}},
        },
    ]
    paths = _paths("tools", "definition")
    leaves = {
        v for d in definitions for p in paths for _, v in iter_concrete_leaves(d, p)
    }
    assert leaves == {"Bearer canary", "canary-2"}


def test_every_call_event_sink_secret_is_registered():
    from api.services.integrations import all_packages
    from api.services.security.secret_surfaces import SURFACES

    declared = {
        f
        for p in all_packages()
        if p.call_event_sink
        for f in p.call_event_sink.sensitive_fields
    }
    assert declared  # bigquery's private_key today
    registered = {s.json_path for s in SURFACES if s.row_key == "CALL_EVENTS"}
    assert {("config", field) for field in declared} <= registered
