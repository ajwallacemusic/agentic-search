"""The FastAPI application: profiles, one-shot search, and SSE-streamed search."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable

import anyio
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response, StreamingResponse

from agentic_search import __version__
from agentic_search.core.harness import Harness, HarnessError
from agentic_search.core.secrets import scrub
from agentic_search.server.config import Profile, ServiceConfig, check_profiles
from agentic_search.server.guard import UNAUTHORIZED, RequestGuard, key_matches
from agentic_search.server.models import (
    RequestError,
    SearchRequest,
    StreamRequest,
    budget_ceilings,
    build_query,
    lean_result,
    render_event,
    resolve_budget,
)
from agentic_search.server.sse import sse_body, wait_for_disconnect

logger = logging.getLogger("agentic_search.server")


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

    close_timeout_s = 10.0  # bound on the shielded close, so a stuck body cannot pin the slot

    def __init__(self, *args: Any, release: Callable[[], None], **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._release = release

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                with anyio.move_on_after(self.close_timeout_s, shield=True) as bounded:
                    await self.body_iterator.aclose()
                if bounded.cancelled_caught:
                    logger.warning("closing a search stream took over %ss; releasing its slot",
                                   self.close_timeout_s)
            finally:
                self._release()


def _describe(outcome: object) -> str:
    """A scrubbed one-line description of a setup error string or an exception."""
    if isinstance(outcome, BaseException):
        return scrub(f"{type(outcome).__name__}: {outcome}")
    return scrub(str(outcome))


def create_app(config: ServiceConfig, profiles: dict[str, Profile | Harness]) -> FastAPI:
    """Build the service. Profiles are set up at startup (and lazily on first use, so the app
    also works without lifespan events) and closed at shutdown."""
    check_profiles(config, profiles)
    profs = {n: p if isinstance(p, Profile) else Profile(harness=p) for n, p in profiles.items()}
    keys = [k.encode() for k in config.auth.keys()]
    gate = _Gate(config.max_concurrent_searches)

    # A failed setup is remembered per profile for `setup_retry_s`, so requests in that window
    # get 503 at once instead of each re-running discovery; afterwards one request retries.
    failures: dict[str, tuple[float, str]] = {}
    setup_locks = {name: asyncio.Lock() for name in profs}

    async def _setup(name: str) -> str | None:
        """Set a profile up if needed; return its scrubbed error if it is unavailable."""
        async with setup_locks[name]:
            failed = failures.get(name)
            if failed is not None and time.monotonic() - failed[0] < config.setup_retry_s:
                return failed[1]
            try:
                await profs[name].harness.setup()
            except HarnessError as exc:
                failures[name] = (time.monotonic(), scrub(str(exc)))
                return failures[name][1]
            failures.pop(name, None)
            return None

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        outcomes = await asyncio.gather(*(_setup(n) for n in profs), return_exceptions=True)
        for name, outcome in zip(profs, outcomes):
            if outcome is not None:
                logger.warning("profile %r is unavailable: %s", name, _describe(outcome))
        yield
        outcomes = await asyncio.gather(*(p.harness.close() for p in profs.values()),
                                        return_exceptions=True)
        for name, outcome in zip(profs, outcomes):
            if isinstance(outcome, BaseException):
                logger.warning("closing profile %r failed: %s", name, _describe(outcome))

    docs: dict[str, Any] = {} if config.expose_docs else {
        "docs_url": None, "redoc_url": None, "openapi_url": None}
    app = FastAPI(title="agentic-search", version=__version__, lifespan=lifespan, **docs)
    # Added before CORS, so CORS is the outer layer and its headers reach 401/413 answers too.
    app.add_middleware(RequestGuard, keys=None if config.auth.type == "none" else keys,
                       max_body_bytes=config.body_limit())
    if config.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=config.cors_origins,
                           allow_methods=["GET", "POST"],
                           allow_headers=["Authorization", "Content-Type", "X-API-Key"],
                           expose_headers=["X-Search-Profile"])

    async def require_key(authorization: str | None = Header(default=None),
                          x_api_key: str | None = Header(default=None)) -> None:
        # RequestGuard already checked the key before the body was read; this stays as a
        # second line of defence for the routes themselves.
        if config.auth.type == "none":
            return
        if not key_matches(keys, authorization, x_api_key):
            raise HTTPException(401, UNAUTHORIZED, headers={"WWW-Authenticate": "Bearer"})

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

    async def ready(name: str) -> None:
        error = await _setup(name)
        if error is not None:
            raise HTTPException(503, f"profile {name!r} is unavailable: {error}")

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
                                         budget_ceilings(profile.harness.budget,
                                                         profile.limits.max_budget)),
                "question": query}

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        ok = all(p.harness.manifests for p in profs.values())
        return {"status": "ok" if ok else "degraded", "version": __version__}

    @app.get("/v1/profiles", dependencies=[Depends(require_key)])
    async def list_profiles() -> dict[str, Any]:
        out = []
        for name, p in profs.items():
            error = await _setup(name)
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

    @app.post("/v1/search", dependencies=[Depends(require_key)], response_model=None)
    async def search(req: SearchRequest, request: Request) -> dict[str, Any] | Response:
        name, profile = resolve(req.profile)
        opts = options(req, profile)
        await ready(name)
        if not gate.try_acquire():
            raise HTTPException(429, "too many concurrent searches")
        task = asyncio.ensure_future(profile.harness.search(opts.pop("question"), **opts))
        watcher = asyncio.ensure_future(wait_for_disconnect(request.receive))
        try:
            await asyncio.wait({task, watcher}, return_when=asyncio.FIRST_COMPLETED)
            if not task.done():
                # The client went away: cancel the search and its in-flight calls. 499 is
                # nginx's "client closed request"; nobody is left to read it.
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                return Response(status_code=499)
            result = task.result()
        except HarnessError as exc:
            raise HTTPException(400, scrub(str(exc))) from exc
        finally:
            watcher.cancel()
            if watcher.done() and not watcher.cancelled():
                watcher.exception()
            if not task.done():  # this handler itself was cancelled
                task.cancel()
                with anyio.CancelScope(shield=True):
                    await asyncio.wait({task})  # hold the slot until the search has stopped
                if not task.cancelled():
                    task.exception()  # mark retrieved; nobody is waiting for this result
            gate.release()
        return lean_result(result, include_content=req.include_content,
                           include_trace=req.include_trace)

    @app.post("/v1/search/stream", dependencies=[Depends(require_key)])
    async def search_stream(req: StreamRequest, request: Request) -> StreamingResponse:
        name, profile = resolve(req.profile)
        opts = options(req, profile)
        await ready(name)
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
