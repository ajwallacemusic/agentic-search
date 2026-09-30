"""Credentials for remote embedders and models, kept separate from the endpoint adapters.

Every provider returns request headers and registers the secret it handles with
core.secrets so it is masked in errors and traces."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Protocol

from agentic_search.core.secrets import register_secret
from agentic_search.embedders.base import EmbedderError

GCP_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_REFRESH_MARGIN_S = 60


class Auth(Protocol):
    async def headers(self) -> dict[str, str]: ...


class NoAuth:
    async def headers(self) -> dict[str, str]:
        return {}


class ApiKey:
    """A static key sent in a header, e.g. `x-api-key: ...` or `api-key: ...` (Azure OpenAI)."""

    def __init__(self, key: str, *, header: str = "x-api-key", prefix: str = ""):
        if not key:
            raise ValueError("ApiKey needs a non-empty key")
        register_secret(key)
        self._value = f"{prefix}{key}"
        self.header = header

    async def headers(self) -> dict[str, str]:
        return {self.header: self._value}


class Bearer:
    def __init__(self, token: str):
        if not token:
            raise ValueError("Bearer needs a non-empty token")
        register_secret(token)
        self._token = token

    async def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"}


class GcpAdc:
    """Google Application Default Credentials (service account, workload identity, gcloud login).
    Needs the `gcp` extra unless `credentials` (a google.auth Credentials object) is injected."""

    def __init__(self, *, scopes: tuple[str, ...] = (GCP_SCOPE,), credentials: Any = None):
        self.scopes = scopes
        self._credentials = credentials
        self._lock = asyncio.Lock()

    def _load_and_refresh(self) -> str:
        try:
            import google.auth
            import google.auth.transport.requests
        except ImportError as exc:
            raise EmbedderError("GcpAdc needs the `gcp` extra: pip install 'agentic-search[gcp]'") from exc
        if self._credentials is None:
            self._credentials, _ = google.auth.default(scopes=list(self.scopes))
        if not self._credentials.valid:
            self._credentials.refresh(google.auth.transport.requests.Request())
        return self._credentials.token

    async def headers(self) -> dict[str, str]:
        async with self._lock:
            try:
                token = await asyncio.to_thread(self._load_and_refresh)
            except EmbedderError:
                raise
            except Exception as exc:
                raise EmbedderError(f"GCP credentials unavailable: {type(exc).__name__}: {exc}") from exc
        register_secret(token)
        return {"Authorization": f"Bearer {token}"}


class AzureIdentity:
    """Microsoft Entra ID token (managed identity, workload identity, az login) for `scope`,
    e.g. "api://<app-id>/.default" or "https://cognitiveservices.azure.com/.default".
    Needs the `azure` extra unless `credential` (an async azure.core TokenCredential) is injected."""

    def __init__(self, scope: str, *, credential: Any = None):
        if not scope:
            raise ValueError("AzureIdentity needs a scope")
        self.scope = scope
        self._credential = credential
        self._token: str | None = None
        self._expires_on = 0.0
        self._lock = asyncio.Lock()

    async def headers(self) -> dict[str, str]:
        async with self._lock:
            if self._token is None or time.time() >= self._expires_on - _REFRESH_MARGIN_S:
                if self._credential is None:
                    try:
                        from azure.identity.aio import DefaultAzureCredential
                    except ImportError as exc:
                        raise EmbedderError(
                            "AzureIdentity needs the `azure` extra: pip install 'agentic-search[azure]'"
                        ) from exc
                    self._credential = DefaultAzureCredential()
                try:
                    access = await self._credential.get_token(self.scope)
                except Exception as exc:
                    raise EmbedderError(f"Azure credentials unavailable: {type(exc).__name__}: {exc}") from exc
                self._token, self._expires_on = access.token, float(access.expires_on)
                register_secret(self._token)
        return {"Authorization": f"Bearer {self._token}"}

    async def close(self) -> None:
        if self._credential is not None and hasattr(self._credential, "close"):
            await self._credential.close()


def build_auth(cfg: dict[str, Any] | None) -> Any:
    """Config form: {type: none|api_key|bearer|gcp_adc|azure_identity, ...}."""
    if not cfg:
        return NoAuth()
    kind = cfg.get("type", "none")
    if kind == "none":
        return NoAuth()
    if kind == "api_key":
        return ApiKey(cfg["key"], header=cfg.get("header", "x-api-key"), prefix=cfg.get("prefix", ""))
    if kind == "bearer":
        return Bearer(cfg["token"])
    if kind == "gcp_adc":
        return GcpAdc(scopes=tuple(cfg.get("scopes", (GCP_SCOPE,))))
    if kind == "azure_identity":
        return AzureIdentity(cfg["scope"])
    raise ValueError(f"unknown auth type {kind!r}; one of none, api_key, bearer, gcp_adc, azure_identity")
