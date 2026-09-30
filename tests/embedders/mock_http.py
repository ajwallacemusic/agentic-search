"""httpx.MockTransport helpers for remote-embedder and model tests (no network)."""

import json

import httpx


def mock(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class Recorder:
    def __init__(self, respond):
        self.requests: list[httpx.Request] = []
        self.respond = respond

    def __call__(self, request):
        self.requests.append(request)
        return self.respond(request)

    def body(self, i=0):
        return json.loads(self.requests[i].content)
