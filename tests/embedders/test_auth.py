import time
from types import SimpleNamespace

import pytest

from agentic_search.core.secrets import scrub
from agentic_search.embedders.auth import (
    ApiKey,
    AzureIdentity,
    Bearer,
    GcpAdc,
    NoAuth,
    build_auth,
)
from agentic_search.embedders.base import EmbedderError


async def test_static_auth_headers_and_registration():
    assert await NoAuth().headers() == {}
    assert await ApiKey("key-abc-123", header="api-key").headers() == {"api-key": "key-abc-123"}
    assert await Bearer("tok-xyz-789").headers() == {"Authorization": "Bearer tok-xyz-789"}
    assert scrub("key-abc-123 tok-xyz-789") == "*** ***"
    with pytest.raises(ValueError):
        ApiKey("")


class FakeGoogleCreds:
    def __init__(self):
        self.valid = False
        self.token = None
        self.refreshes = 0

    def refresh(self, request):
        self.refreshes += 1
        self.token = f"ya29.fake-{self.refreshes}"
        self.valid = True


async def test_gcp_adc_refreshes_only_when_invalid():
    creds = FakeGoogleCreds()
    auth = GcpAdc(credentials=creds)
    assert await auth.headers() == {"Authorization": "Bearer ya29.fake-1"}
    assert await auth.headers() == {"Authorization": "Bearer ya29.fake-1"}
    assert creds.refreshes == 1
    creds.valid = False
    assert (await auth.headers())["Authorization"] == "Bearer ya29.fake-2"
    assert scrub("ya29.fake-2") == "***"


async def test_gcp_adc_failure_is_embedder_error():
    class Broken(FakeGoogleCreds):
        def refresh(self, request):
            raise RuntimeError("metadata server unreachable")

    with pytest.raises(EmbedderError, match="metadata server unreachable"):
        await GcpAdc(credentials=Broken()).headers()


class FakeAzureCred:
    def __init__(self, ttl=3600):
        self.calls = 0
        self.ttl = ttl
        self.closed = False

    async def get_token(self, scope):
        self.calls += 1
        return SimpleNamespace(token=f"eyJ.fake-{self.calls}", expires_on=time.time() + self.ttl)

    async def close(self):
        self.closed = True


async def test_azure_identity_caches_until_near_expiry():
    cred = FakeAzureCred()
    auth = AzureIdentity("api://medsiglip/.default", credential=cred)
    assert await auth.headers() == {"Authorization": "Bearer eyJ.fake-1"}
    await auth.headers()
    assert cred.calls == 1
    short = FakeAzureCred(ttl=30)  # inside the 60 s refresh margin
    auth2 = AzureIdentity("s", credential=short)
    await auth2.headers()
    await auth2.headers()
    assert short.calls == 2
    await auth.close()
    assert cred.closed and scrub("eyJ.fake-1") == "***"


async def test_azure_failure_is_embedder_error():
    class Broken(FakeAzureCred):
        async def get_token(self, scope):
            raise RuntimeError("no managed identity")

    with pytest.raises(EmbedderError, match="no managed identity"):
        await AzureIdentity("s", credential=Broken()).headers()


def test_build_auth():
    assert isinstance(build_auth(None), NoAuth)
    assert isinstance(build_auth({"type": "bearer", "token": "abcd-token"}), Bearer)
    assert isinstance(build_auth({"type": "api_key", "key": "k-1234"}), ApiKey)
    assert isinstance(build_auth({"type": "gcp_adc"}), GcpAdc)
    assert isinstance(build_auth({"type": "azure_identity", "scope": "s"}), AzureIdentity)
    with pytest.raises(ValueError):
        build_auth({"type": "kerberos"})
