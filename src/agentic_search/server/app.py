"""The FastAPI application: profiles, one-shot search, and SSE-streamed search."""

from __future__ import annotations

import asyncio
import hmac
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable

import anyio
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

from agentic_search import __version__
from agentic_search.core.harness import Harness, HarnessError
from agentic_search.core.secrets import scrub
from agentic_search.server.config import Profile, ServiceConfig, check_profiles
from agentic_search.server.models import (
    RequestError,
    SearchRequest,
    StreamRequest,
    build_query,
    lean_result,
    render_event,
    resolve_budget,
)
from agentic_search.server.sse import sse_body


class _Gate:
    """Non-blocking concurrency limit: a search either gets a slot now or the caller gets 429."""

    def __init__(self, limit: int):
        self.limit, self.active = limit, 0

    def try_acquire(self) -> bool:
        if self.active >= self.limit:
            return False
        self.active += 1
        return True

    def release(self) -> None:
        self.active -= 1


class _GatedStreamingResponse(StreamingResponse):
    """Releases the search slot however the response ends, even if the body never starts."""

    def __init__(self, *args: Any, release: Callable[[], None], **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._release = release

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                with anyio.CancelScope(shield=True):
                    await self.body_iterator.aclose()
            finally:
                self._release()


def create_app(config: ServiceConfig, profiles: dict[str, Profile | Harness]) -> FastAPI:
    """Build the service. Profiles are set up at startup (and lazily on first use, so the app
    also works without lifespan events) and closed at shutdown."""
    check_profiles(config, profiles)
    profs = {n: p if isinstance(p, Profile) else Profile(harness=p) for n, p in profiles.items()}
    keys = [k.encode() for k in config.auth.keys()]
    gate = _Gate(config.max_concurrent_searches)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await asyncio.gather(*(_setup(p) for p in profs.values()), return_exceptions=True)
        yield
        await asyncio.gather(*(p.harness.close() for p in profs.values()), return_exceptions=True)

    app = FastAPI(title="agentic-search", version=__version__, lifespan=lifespan)
    if config.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=config.cors_origins,
                           allow_methods=["GET", "POST"],
                           allow_headers=["Authorization", "Content-Type", "X-API-Key"],
                           expose_headers=["X-Search-Profile"])

    async def require_key(authorization: str | None = Header(default=None),
                          x_api_key: str | None = Header(default=None)) -> None:
        if config.auth.type == "none":
            return
        token = x_api_key
        if authorization and authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        candidate = (token or "").encode("utf-8", "surrogateescape")
        if not token or not any(hmac.compare_digest(candidate, k) for k in keys):
            raise HTTPException(401, "missing or invalid API key",
                                headers={"WWW-Authenticate": "Bearer"})

    def resolve(name: str | None) -> tuple[str, Profile]:
        if name is None:
            if config.default_profile is not None:
                name = config.default_profile
            elif len(profs) == 1:
                name = next(iter(profs))
            else:
                raise HTTPException(400, f"profile is required; one of {sorted(profs)}")
        if name not in profs:
            raise HTTPException(404, f"unknown profile {name!r}")
        return name, profs[name]

    async def _setup(profile: Profile) -> None:
        await profile.harness.setup()

    async def ready(name: str, profile: Profile) -> None:
        try:
            await _setup(profile)
        except HarnessError as exc:
            raise HTTPException(503, f"profile {name!r} is unavailable: {scrub(str(exc))}") from exc

    def options(req: SearchRequest, profile: Profile) -> dict[str, Any]:
        if req.include_content and not profile.limits.allow_include_content:
            raise HTTPException(400, "this profile does not allow include_content")
        try:
            query = build_query(req, max_images=config.max_images,
                                max_image_bytes=config.max_image_bytes)
        except RequestError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"sources": req.sources, "top_k": req.top_k, "mode": req.mode,
                "budget": resolve_budget(profile.harness.budget, req.budget,
                                         profile.limits.max_budget),
                "question": query}

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        ok = all(p.harness.manifests for p in profs.values())
        return {"status": "ok" if ok else "degraded", "version": __version__}

    @app.get("/v1/profiles", dependencies=[Depends(require_key)])
    async def list_profiles() -> dict[str, Any]:
        out = []
        for name, p in profs.items():
            error = None
            try:
                await _setup(p)
            except HarnessError as exc:
                error = scrub(str(exc))
            h = p.harness
            out.append({
                "name": name,
                "default": name == config.default_profile,
                "available": error is None,
                "error": error,
                "mode": h.mode,
                "budget": h.budget.model_dump(mode="json"),
                "limits": p.limits.model_dump(mode="json"),
                "sources": [{"name": m.source, "backend_type": m.backend_type,
                             "capabilities": sorted(c.value for c in m.capabilities),
                             "collections": [c.name for c in m.collections],
                             "description": m.description}
                            for m in sorted(h.manifests.values(), key=lambda m: m.source)],
                "setup_errors": dict(h.setup_errors),
            })
        return {"profiles": out}

    @app.post("/v1/search", dependencies=[Depends(require_key)])
    async def search(req: SearchRequest) -> dict[str, Any]:
        name, profile = resolve(req.profile)
        opts = options(req, profile)
        await ready(name, profile)
        if not gate.try_acquire():
            raise HTTPException(429, "too many concurrent searches")
        try:
            result = await profile.harness.search(opts.pop("question"), **opts)
        except HarnessError as exc:
            raise HTTPException(400, scrub(str(exc))) from exc
        finally:
            gate.release()
        return lean_result(result, include_content=req.include_content,
                           include_trace=req.include_trace)

    @app.post("/v1/search/stream", dependencies=[Depends(require_key)])
    async def search_stream(req: StreamRequest, request: Request) -> StreamingResponse:
        name, profile = resolve(req.profile)
        opts = options(req, profile)
        await ready(name, profile)
        try:
            stream = profile.harness.stream(opts.pop("question"), snapshot_k=req.snapshot_k,
                                            include_content=req.include_content, **opts)
        except HarnessError as exc:
            raise HTTPException(400, scrub(str(exc))) from exc

        def render(event: Any) -> str:
            return render_event(event, include_content=req.include_content,
                                include_trace=req.include_trace)

        body = sse_body(stream, render, keepalive_s=config.keepalive_s, receive=request.receive)
        response = _GatedStreamingResponse(
            body, release=gate.release, media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                     "X-Search-Profile": name})
        if not gate.try_acquire():
            raise HTTPException(429, "too many concurrent searches")
        return response

    return app
