from types import SimpleNamespace

import pytest

from agentic_search.embedders.auth import GcpAdc
from agentic_search.models.llm import ChatMessage
from agentic_search.models.vertex import vertex_base_url, vertex_client


class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        msg = SimpleNamespace(content="ok", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1))


class FixedAuth:
    async def headers(self):
        return {"Authorization": "Bearer ya29.test"}


def test_global_endpoint_has_no_region_prefix():
    assert vertex_base_url("my-project-1") == (
        "https://aiplatform.googleapis.com/v1/projects/my-project-1/locations/global/endpoints/openapi")


def test_regional_endpoint_uses_the_region_host():
    assert vertex_base_url("my-project-1", "us-central1") == (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project-1"
        "/locations/us-central1/endpoints/openapi")


@pytest.mark.parametrize("project,location", [("Bad_Project", "global"), ("my-project-1", "us central1"),
                                              ("my-project-1", "../x")])
def test_rejects_malformed_project_or_location(project, location):
    with pytest.raises(ValueError):
        vertex_base_url(project, location)


async def test_client_sends_publisher_model_and_auth_header():
    completions = FakeCompletions()
    fake = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    c = vertex_client("gemini-3.8-flash", project="my-project-1", auth=FixedAuth(), client=fake)
    assert c.id == "vertex:gemini-3.8-flash"
    assert c.supports_images is True
    r = await c.chat("s", [ChatMessage(role="user", content="q")])
    assert r.text == "ok"
    assert completions.kwargs["model"] == "google/gemini-3.8-flash"
    assert completions.kwargs["extra_headers"] == {"Authorization": "Bearer ya29.test"}


def test_publisher_prefix_is_kept():
    fake = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    c = vertex_client("meta/llama-4", project="my-project-1", auth=FixedAuth(), client=fake)
    assert c.model == "meta/llama-4" and c.id == "vertex:meta/llama-4"


def test_default_auth_is_application_default_credentials():
    fake = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    c = vertex_client("gemini-3.8-flash", project="my-project-1", client=fake)
    assert isinstance(c.auth, GcpAdc)
