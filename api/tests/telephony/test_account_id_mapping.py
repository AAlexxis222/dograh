"""The account-id mapping is a fact about a provider: it lives on the provider registry.

The db clients use it to fill ``telephony_configurations.account_id``; a process that
imports only a db client must still get the right value (no silent NULL).
"""

import os
import subprocess
import sys

import pytest

from api.services.telephony import registry
from api.services.telephony.registry import account_id_for
from api.utils.stored_credentials import credential_value


@pytest.mark.parametrize(
    "credentials, expected",
    [
        ({"f": "v"}, "v"),
        ({"f": 0}, 0),
        ({"f": False}, False),
        ({"f": ["x"]}, ["x"]),
        ({"f": ""}, None),
        ({"f": None}, None),
        ({}, None),
        (None, None),
        ('{"f": "v"}', None),  # a legacy document stored double-encoded is a string
        (["f"], None),
    ],
)
def test_credential_value_reads_a_field_of_an_object_document(credentials, expected):
    assert credential_value(credentials, "f") == expected


@pytest.mark.parametrize(
    "provider, credentials, expected",
    [
        ("twilio", {"account_sid": "AC1"}, "AC1"),
        ("vonage", {"api_key": "vk1", "api_secret": "s"}, "vk1"),
        ("telnyx", {"api_key": "SECRET", "connection_id": "c1"}, "c1"),
        ("exotel", {"api_key": "SECRET", "account_sid": "e1"}, "e1"),
        ("plivo", {"auth_id": "p1"}, "p1"),
        ("vobiz", {"auth_id": "b1"}, "b1"),
        ("cloudonix", {"domain_id": "acme.cx.net"}, "acme.cx.net"),
        ("twilio", {"account_sid": 12345}, "12345"),
        ("twilio", {"account_sid": True}, None),
        ("twilio", {"account_sid": ["AC1"]}, None),
        ("twilio", {"account_sid": ""}, None),
        ("twilio", "not an object", None),
        ("ari", {"app_name": "dograh", "account_sid": "x"}, None),
        ("not-a-provider", {"account_sid": "AC1"}, None),
    ],
)
def test_the_provider_spec_maps_its_credentials_to_the_account_id(
    provider, credentials, expected
):
    assert account_id_for(provider, credentials) == expected
    spec = registry.get_optional(provider)
    if spec is not None:
        assert spec.account_id_from(credentials) == expected


def test_a_process_that_imports_only_the_db_client_maps_account_ids():
    code = (
        "from api.db.telephony_configuration_client import account_id_for;"
        "print(account_id_for('twilio', {'account_sid': 'AC1'}),"
        " account_id_for('vonage', {'api_key': 'vk1'}))"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().splitlines()[-1] == "AC1 vk1"


def test_the_db_layer_does_not_break_the_provider_db_client_import():
    """Importing the provider registry at module level from a db client makes
    ``from api.db import db_client`` inside a provider return the ``api.db.db_client``
    module instead of the client, when ``api.db`` is imported first."""
    code = (
        "import api.db;"
        "import api.services.telephony.factory as factory;"
        "print(type(factory.db_client).__name__)"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().splitlines()[-1] == "DBClient"
