"""Same contract as test_gmail_token_persist, for the other Google clients:
a read-only credentials mount (EROFS) must not kill a successful refresh."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.tools.google_calendar import GoogleCalendarClient
from core.tools.google_tasks import GoogleTasksClient

_EROFS = OSError(30, "Read-only file system")
_CLIENTS = [GoogleCalendarClient, GoogleTasksClient]


@pytest.mark.parametrize("client_cls", _CLIENTS)
def test_persist_token_swallows_erofs(client_cls, tmp_path: Path) -> None:
    client = client_cls(token_path=tmp_path / "store" / "token.json")
    creds = MagicMock()
    creds.to_json.return_value = "{}"
    with patch.object(Path, "write_text", side_effect=_EROFS):
        client._persist_token(creds)  # must not raise


@pytest.mark.parametrize("client_cls", _CLIENTS)
def test_get_credentials_survives_readonly_token_store(client_cls, tmp_path: Path) -> None:
    pytest.importorskip("google.oauth2.credentials")
    client = client_cls(token_path=tmp_path / "store" / "token.json")
    client.token_path.parent.mkdir(parents=True)
    client.token_path.write_text("{}")

    creds = MagicMock()
    creds.valid = False
    creds.expired = True
    creds.refresh_token = "present"

    with (
        patch("google.oauth2.credentials.Credentials") as cred_cls,
        patch("google.auth.transport.requests.Request"),
        patch.object(Path, "write_text", side_effect=_EROFS),
    ):
        cred_cls.from_authorized_user_file.return_value = creds
        assert client._get_credentials() is creds

    creds.refresh.assert_called_once()
