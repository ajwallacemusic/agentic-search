"""A pure-ASGI guard for `/v1/*`: API-key check before any body is read, and a body size cap."""

from __future__ import annotations

import hmac
from typing import Any

from starlette.responses import JSONResponse

UNAUTHORIZED = "missing or invalid API key"
TOO_LARGE = "request body too large"


def key_matches(keys: list[bytes], authorization: str | None, x_api_key: str | None) -> bool:
    """`Authorization: Bearer <key>` wins over `X-API-Key`; compared in constant time."""
    token = x_api_key
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    candidate = (token or "").encode("utf-8", "surrogateescape")
    return bool(token) and any(hmac.compare_digest(candidate, k) for k in keys)


class RequestGuard:
    """For `/v1/*` requests: refuse a bad key with `401` before reading the body, then read the
    body up to `max_body_bytes` (refusing a larger `Content-Length`, or a streamed body once it
    grows past the cap, with `413`) and replay it to the app. `keys=None` turns auth off."""

    def __init__(self, app: Any, *, keys: list[bytes] | None, max_body_bytes: int,
                 prefix: str = "/v1/"):
        self.app, self.keys, self.max_body_bytes, self.prefix = app, keys, max_body_bytes, prefix

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or not scope["path"].startswith(self.prefix):
            await self.app(scope, receive, send)
            return
        headers: dict[bytes, str] = {}
        for name, value in scope["headers"]:
            headers.setdefault(name.lower(), value.decode("latin-1"))
        if self.keys is not None and not key_matches(
                self.keys, headers.get(b"authorization"), headers.get(b"x-api-key")):
            await JSONResponse({"detail": UNAUTHORIZED}, 401,
                               headers={"WWW-Authenticate": "Bearer"})(scope, receive, send)
            return
        declared = headers.get(b"content-length", "").strip()
        if declared.isdigit() and int(declared) > self.max_body_bytes:
            await JSONResponse({"detail": TOO_LARGE}, 413)(scope, receive, send)
            return
        chunks: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return  # the client left before sending its body: nothing to answer
            body = message.get("body", b"")
            size += len(body)
            if size > self.max_body_bytes:
                await JSONResponse({"detail": TOO_LARGE}, 413)(scope, receive, send)
                return
            chunks.append(body)
            if not message.get("more_body", False):
                break
        replayed = False

        async def replay() -> dict[str, Any]:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()  # afterwards only http.disconnect: streams watch for it

        await self.app(scope, replay, send)
