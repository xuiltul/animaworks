from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Minimal Twilio REST client and request-signature verification."""

import base64
import hashlib
import hmac
import logging
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote, urlencode, urlsplit, urlunsplit

import httpx

from core.config.schemas import PhoneConfig

logger = logging.getLogger(__name__)

_TWILIO_API_BASE = "https://api.twilio.com/2010-04-01/Accounts"


class TwilioAPIError(RuntimeError):
    """Safe, credential-free Twilio API failure."""


class TwilioClient:
    """Small async wrapper around the Twilio Calls REST resource."""

    def __init__(
        self,
        account_sid: str,
        auth_token: str,
        *,
        api_base_url: str = _TWILIO_API_BASE,
        timeout: float = 15.0,
    ) -> None:
        if not account_sid or not auth_token:
            raise TwilioAPIError("Twilio credentials are not configured")
        self._account_sid = account_sid
        self._auth_token = auth_token
        self._api_base_url = api_base_url.rstrip("/")
        self._timeout = timeout

    @property
    def _calls_url(self) -> str:
        return f"{self._api_base_url}/{quote(self._account_sid, safe='')}/Calls.json"

    async def create_call(
        self,
        to: str,
        from_: str,
        url: str,
        status_callback: str,
    ) -> dict[str, Any]:
        """Create a call and return Twilio's response object."""
        payload = {
            "To": to,
            "From": from_,
            "Url": url,
            "Method": "POST",
            "StatusCallback": status_callback,
            "StatusCallbackMethod": "POST",
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    self._calls_url,
                    data=payload,
                    auth=httpx.BasicAuth(self._account_sid, self._auth_token),
                )
        except httpx.HTTPError:
            raise TwilioAPIError("Twilio call request failed") from None

        if response.status_code < 200 or response.status_code >= 300:
            raise TwilioAPIError(f"Twilio call request returned HTTP {response.status_code}")
        try:
            data = response.json()
        except (ValueError, TypeError):
            raise TwilioAPIError("Twilio returned an invalid call response") from None
        if not isinstance(data, dict) or not data.get("sid"):
            raise TwilioAPIError("Twilio response did not contain a call SID")
        return data

    async def get_call(self, sid: str) -> dict[str, Any]:
        """Fetch one call resource by SID."""
        call_url = f"{self._api_base_url}/{quote(self._account_sid, safe='')}/Calls/{quote(sid, safe='')}.json"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(
                    call_url,
                    auth=httpx.BasicAuth(self._account_sid, self._auth_token),
                )
        except httpx.HTTPError:
            raise TwilioAPIError("Twilio call lookup failed") from None

        if response.status_code < 200 or response.status_code >= 300:
            raise TwilioAPIError(f"Twilio call lookup returned HTTP {response.status_code}")
        try:
            data = response.json()
        except (ValueError, TypeError):
            raise TwilioAPIError("Twilio returned an invalid call response") from None
        if not isinstance(data, dict):
            raise TwilioAPIError("Twilio returned an invalid call response")
        return data


def get_twilio_credentials(config: PhoneConfig) -> tuple[str | None, str | None]:
    """Read account SID and auth token from the configured shared vault keys."""
    try:
        from core.config.vault import get_vault_manager

        vault = get_vault_manager()
        account_sid = vault.get("shared", config.account_sid_vault_key)
        auth_token = vault.get("shared", config.auth_token_vault_key)
    except Exception:
        # Never include vault errors or values in logs/exceptions at the webhook boundary.
        logger.warning("Unable to read Twilio credentials from the shared vault")
        return None, None
    return account_sid, auth_token


def create_twilio_client(config: PhoneConfig) -> TwilioClient:
    """Build a REST client from the configured shared vault credentials."""
    account_sid, auth_token = get_twilio_credentials(config)
    if not account_sid or not auth_token:
        raise TwilioAPIError("Twilio credentials are not configured")
    return TwilioClient(account_sid, auth_token)


def build_webhook_url(
    public_base_url: str,
    path: str,
    *,
    params: Mapping[str, str | int] | None = None,
    raw_query: str | None = None,
) -> str:
    """Build a public webhook URL without trusting the proxy-facing request URL."""
    base = public_base_url.strip().rstrip("/")
    parsed = urlsplit(base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("phone.public_base_url must be an absolute HTTP(S) URL")
    normalized_path = path if path.startswith("/") else f"/{path}"
    url = f"{base}{normalized_path}"
    if raw_query:
        return f"{url}?{raw_query}"
    if params:
        return f"{url}?{urlencode(params)}"
    return url


def _url_port_variants(url: str) -> list[str]:
    """Return *url* as given, without its port, and with its port spelled out."""
    parts = urlsplit(url)
    if not parts.netloc:
        return [url]
    try:
        explicit_port = parts.port
    except ValueError:
        return [url]
    host = parts.netloc.rsplit(":", 1)[0] if explicit_port else parts.netloc
    port = explicit_port or (443 if parts.scheme == "https" else 80)
    return list(
        dict.fromkeys(
            (
                url,
                urlunsplit(parts._replace(netloc=host)),
                urlunsplit(parts._replace(netloc=f"{host}:{port}")),
            )
        )
    )


def validate_signature(
    auth_token: str,
    url: str,
    params: Mapping[str, str | Sequence[str]],
    signature: str,
) -> bool:
    """Validate Twilio's X-Twilio-Signature (HMAC-SHA1 + Base64).

    Twilio's signature input is the exact externally visible URL followed by
    each POST parameter name and value in lexicographic parameter-name order.
    Multi-value parameters are appended in sorted value order as in Twilio's
    request-validator implementations.

    Twilio does not sign the port consistently: on a non-default port its voice
    webhook and status callback can differ. Like its own validators, this accepts
    the URL with or without the port.
    """
    if not auth_token or not signature:
        return False

    signed_params = ""
    for key in sorted(params):
        raw_value = params[key]
        values = raw_value if isinstance(raw_value, Sequence) and not isinstance(raw_value, str) else [raw_value]
        for value in sorted(str(item) for item in values):
            signed_params += str(key) + value

    try:
        return any(
            hmac.compare_digest(
                base64.b64encode(
                    hmac.new(
                        auth_token.encode("utf-8"), (candidate + signed_params).encode("utf-8"), hashlib.sha1
                    ).digest()
                ).decode("ascii"),
                signature,
            )
            for candidate in _url_port_variants(url)
        )
    except TypeError:
        return False


__all__ = [
    "TwilioAPIError",
    "TwilioClient",
    "build_webhook_url",
    "create_twilio_client",
    "get_twilio_credentials",
    "validate_signature",
]
