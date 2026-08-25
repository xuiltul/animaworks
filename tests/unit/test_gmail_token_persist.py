"""Token persist must be best-effort: sandboxed runs mount the
credentials dir read-only (EROFS), and a failed save must not kill
an otherwise-successful refresh."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from core.tools.gmail import GmailClient


def _client(tmp_path: Path) -> GmailClient:
    return GmailClient(
        token_path=tmp_path / "store" / "token.json",
        mcp_token_path=tmp_path / "absent-mcp-token.json",
    )


_EROFS = OSError(30, "Read-only file system")


def test_persist_token_swallows_erofs(tmp_path: Path) -> None:
    client = _client(tmp_path)
    creds = MagicMock()
    creds.to_json.return_value = "{}"
    with patch.object(Path, "write_text", side_effect=_EROFS):
        client._persist_token(creds)  # must not raise


def test_get_credentials_survives_readonly_token_store(tmp_path: Path) -> None:
    client = _client(tmp_path)
    client.token_path.parent.mkdir(parents=True)
    client.token_path.write_text("{}")

    creds = MagicMock()
    creds.valid = False
    creds.expired = True
    creds.refresh_token = "present"

    with (
        patch("core.tools.gmail.Credentials") as cred_cls,
        patch("core.tools.gmail.Request"),
        patch.object(Path, "write_text", side_effect=_EROFS),
    ):
        cred_cls.from_authorized_user_file.return_value = creds
        assert client._get_credentials() is creds

    creds.refresh.assert_called_once()
