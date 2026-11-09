from agentic_search.core.types import ModelUsage
from agentic_search.models.llm import ChatResponse
from demo.ignite.answer import answer_from, is_correct


class FakeClient:
    id = "fake"

    def __init__(self, text="Richland County."):
        self.text, self.seen = text, None

    async def chat(self, system, messages, **kw):
        self.seen = (system, messages[0].content)
        return ChatResponse(text=self.text, tool_calls=[], usage=ModelUsage(), stop_reason="stop")


def test_is_correct_matches_any_alias_ignoring_case_and_punctuation():
    expected = ["Richland County", "Richland County, South Carolina"]
    assert is_correct("It is in richland county.", expected) is True
    assert is_correct("Lexington County", expected) is False


def test_is_correct_is_none_without_a_known_answer():
    assert is_correct("anything", []) is None


async def test_answer_reads_only_the_given_documents_in_order():
    c = FakeClient()
    docs = [("A title", "first text"), ("B title", "second text")]
    out = await answer_from(c, "Which county?", docs)
    assert out == "Richland County."
    system, prompt = c.seen
    assert "only" in system.lower()
    assert prompt.index("[1] A title") < prompt.index("[2] B title")
    assert "Which county?" in prompt


async def test_answer_without_documents_says_so_without_a_model_call():
    c = FakeClient()
    assert "No documents" in await answer_from(c, "q", [])
    assert c.seen is None
