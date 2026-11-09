from agentic_search.core.types import ModelUsage
from agentic_search.models.llm import ChatResponse
from demo.ignite.answer import NOT_FOUND, answer_from, is_correct, parse_reply


class FakeClient:
    id = "fake"

    def __init__(self, text="[1] fact one\n[2] fact two\nAnswer: Richland County"):
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
    out, steps = await answer_from(c, "Which county?", docs)
    assert out == "Richland County" and steps == ["[1] fact one", "[2] fact two"]
    system, prompt = c.seen
    assert "only" in system.lower()
    assert prompt.index("[1] A title") < prompt.index("[2] B title")
    assert "Which county?" in prompt


async def test_answer_without_documents_says_so_without_a_model_call():
    c = FakeClient()
    assert "No documents" in (await answer_from(c, "q", []))[0]
    assert c.seen is None


def test_parse_reply_takes_the_last_answer_line():
    text = "Step 1 [2]: from South Carolina\n**Answer:** Richland County\n"
    assert parse_reply(text) == ("Richland County", ["Step 1 [2]: from South Carolina"])
    assert parse_reply("Answer: draft\nmore\nanswer: final")[0] == "final"


def test_parse_reply_without_an_answer_line_is_not_found():
    answer, steps = parse_reply("Step 1: Andrew Deveaux was born in South")
    assert answer == NOT_FOUND and steps == ["Step 1: Andrew Deveaux was born in South"]


def test_the_reader_is_told_to_accept_loose_steps():
    from demo.ignite.answer import ANSWER_SYSTEM
    assert "loosely" in ANSWER_SYSTEM and "Answer:" in ANSWER_SYSTEM


async def test_notes_are_sent_with_the_documents():
    c = FakeClient()
    await answer_from(c, "Same wood tone?", [("table", "dswoodtone : light wood")],
                      ["The chair has dswoodtone : light wood."])
    prompt = c.seen[1]
    assert "Notes the search took:" in prompt and "- The chair has dswoodtone" in prompt
    assert prompt.index("[1] table") < prompt.index("Notes the search took:")


async def test_no_notes_section_without_notes():
    c = FakeClient()
    await answer_from(c, "q", [("t", "x")])
    assert "Notes" not in c.seen[1]
