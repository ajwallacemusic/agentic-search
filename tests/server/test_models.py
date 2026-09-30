"""Tests for server request/response models and guard functions."""

import base64
import json

import pytest

from agentic_search.core.result import RankedHit, SearchResult
from agentic_search.core.state import Trace, TraceEvent, Usage
from agentic_search.core.types import Hit, ImagePart, Query, TextPart
from agentic_search.events import PhaseStarted, SearchFinished
from agentic_search.server.models import (
    ImageInput,
    RequestError,
    SearchRequest,
    build_query,
    lean_result,
    render_event,
)


class TestBuildQuery:
    """Tests for build_query guard function."""

    def test_text_plus_standard_base64_image(self):
        """Text plus one standard-base64 image gives a Query with TextPart and ImagePart."""
        raw_data = b"PNG image bytes"
        encoded = base64.b64encode(raw_data).decode()
        req = SearchRequest(
            question="What's this?",
            images=[ImageInput(data=encoded, mime="image/png")]
        )

        query = build_query(req, max_images=4, max_image_bytes=10_000_000)

        assert isinstance(query, Query)
        assert len(query.content) == 2
        assert isinstance(query.content[0], TextPart)
        assert query.content[0].text == "What's this?"
        assert isinstance(query.content[1], ImagePart)
        assert query.content[1].data == raw_data
        assert query.content[1].uri is None
        assert query.content[1].mime == "image/png"

    def test_url_safe_base64_input_decodes(self):
        """URL-safe base64 input (with - and _) decodes correctly."""
        raw_data = b"\xff\xfe\xfd\xfc\xfb\xfa"
        # Standard base64 uses + and /
        standard = base64.b64encode(raw_data).decode()
        # URL-safe base64 uses - and _
        url_safe = standard.replace("+", "-").replace("/", "_")

        req = SearchRequest(
            question="test",
            images=[ImageInput(data=url_safe)]
        )
        query = build_query(req, max_images=1, max_image_bytes=10_000_000)

        assert query.content[1].data == raw_data

    @pytest.mark.parametrize("raw_data", [b"\xfb", b"\xfb\xff"])
    def test_unpadded_url_safe_base64_decodes(self, raw_data):
        """Unpadded URL-safe base64 (1 and 2-byte payloads) decode after adding padding."""
        # 1-byte and 2-byte payloads produce 2 and 3 character base64 respectively,
        # both needing padding to reach a multiple of 4
        url_safe_padded = base64.urlsafe_b64encode(raw_data).decode()

        # Remove padding - this makes the base64 unpadded
        url_safe_unpadded = url_safe_padded.rstrip("=")
        assert len(url_safe_unpadded) % 4 != 0, "Test setup: must need padding"

        req = SearchRequest(
            question="test",
            images=[ImageInput(data=url_safe_unpadded)]
        )
        query = build_query(req, max_images=1, max_image_bytes=10_000_000)

        assert query.content[1].data == raw_data

    def test_empty_image_data_raises_error(self):
        """Empty image data raises RequestError."""
        req = SearchRequest(
            question="test",
            images=[ImageInput(data="")]
        )
        with pytest.raises(RequestError, match="images\\[0\\]\\.data is empty"):
            build_query(req, max_images=1, max_image_bytes=10_000_000)

    def test_too_many_images_raises_error(self):
        """More than max_images raises RequestError."""
        encoded = base64.b64encode(b"x").decode()
        req = SearchRequest(
            question="test",
            images=[
                ImageInput(data=encoded),
                ImageInput(data=encoded),
                ImageInput(data=encoded),
            ]
        )
        with pytest.raises(RequestError, match="at most 2 images"):
            build_query(req, max_images=2, max_image_bytes=10_000_000)

    def test_invalid_base64_raises_error(self):
        """Invalid base64 characters raise RequestError."""
        req = SearchRequest(
            question="test",
            images=[ImageInput(data="@@@")]
        )
        with pytest.raises(RequestError, match="images\\[0\\]\\.data is not valid base64"):
            build_query(req, max_images=1, max_image_bytes=10_000_000)

    @pytest.mark.parametrize("invalid_data", ["A", "AAAAA"])
    def test_invalid_base64_length_raises_error(self, invalid_data):
        """Base64 data with length that is 1 mod 4 raises RequestError."""
        req = SearchRequest(
            question="test",
            images=[ImageInput(data=invalid_data)]
        )
        with pytest.raises(RequestError, match="images\\[0\\]\\.data is not valid base64"):
            build_query(req, max_images=1, max_image_bytes=10_000_000)

    def test_image_larger_than_max_bytes_raises_error(self):
        """Image larger than max_image_bytes raises RequestError after decoding."""
        raw_data = b"x" * 100
        encoded = base64.b64encode(raw_data).decode()
        req = SearchRequest(
            question="test",
            images=[ImageInput(data=encoded)]
        )
        with pytest.raises(RequestError, match="images\\[0\\] is larger than 50 bytes"):
            build_query(req, max_images=1, max_image_bytes=50)

    def test_oversized_encoded_data_rejects_before_decoding(self):
        """Oversized encoded data with invalid chars rejects before decoding attempt."""
        # Use invalid base64 characters (@) and make it exceed the pre-decode threshold.
        # With invalid data, we need to verify the error is from the pre-decode size check
        # ("larger than") not from the base64 validation ("not valid base64").
        max_bytes = 10
        threshold = max_bytes * 4 // 3 + 4
        # Create data that is both oversized AND invalid base64
        oversized_invalid = "@" * (threshold + 10)
        assert len(oversized_invalid) > threshold

        req = SearchRequest(
            question="test",
            images=[ImageInput(data=oversized_invalid)]
        )
        # Pre-decode check should trigger first, producing "larger than" message
        with pytest.raises(RequestError, match="images\\[0\\] is larger than"):
            build_query(req, max_images=1, max_image_bytes=max_bytes)


class TestLeanResult:
    """Tests for lean_result projection function."""

    @pytest.fixture
    def sample_result(self):
        """Create a real SearchResult for testing."""
        hit = Hit(
            source="test_source",
            doc_id="1",
            content=[TextPart(text="This is the full content")],
        )
        ranked_hit = RankedHit(hit=hit, score=0.95)

        trace = Trace(events=[TraceEvent(type="phase_started", turn=1, at_ms=0.0)])
        usage = Usage(input_tokens=10, output_tokens=5)

        result = SearchResult(
            question=Query(content=[TextPart(text="test")]),
            hits=[ranked_hit],
            stop_reason="controller_stop",
            usage=usage,
            trace=trace,
            mode="retrieval",
        )
        return result

    def test_lean_result_excludes_trace_and_content_by_default(self, sample_result):
        """By default, trace and hit content are excluded."""
        projected = lean_result(sample_result, include_content=False, include_trace=False)

        assert "trace" not in projected
        assert "content" not in projected["hits"][0]["hit"]
        # But other fields should be present
        assert "hits" in projected
        assert "stop_reason" in projected

    def test_lean_result_includes_trace_when_requested(self, sample_result):
        """With include_trace=True, trace is included."""
        projected = lean_result(sample_result, include_content=False, include_trace=True)

        assert "trace" in projected
        assert isinstance(projected["trace"], dict)

    def test_lean_result_includes_content_when_requested(self, sample_result):
        """With include_content=True, hit content is included."""
        projected = lean_result(sample_result, include_content=True, include_trace=False)

        assert "trace" not in projected
        assert "content" in projected["hits"][0]["hit"]
        assert len(projected["hits"][0]["hit"]["content"]) == 1
        assert projected["hits"][0]["hit"]["content"][0]["text"] == "This is the full content"


class TestRenderEvent:
    """Tests for render_event serialization."""

    def test_search_finished_is_single_line_json(self):
        """SearchFinished event renders as single-line JSON with lean result."""
        hit = Hit(
            source="test",
            doc_id="1",
            content=[TextPart(text="content")],
        )
        result = SearchResult(
            question=Query(content=[TextPart(text="q")]),
            hits=[RankedHit(hit=hit, score=0.9)],
            stop_reason="controller_stop",
            usage=Usage(input_tokens=1, output_tokens=1),
            trace=Trace(events=[TraceEvent(type="phase_started", turn=1, at_ms=0.0)]),
            mode="retrieval",
        )
        event = SearchFinished(
            search_id="s1",
            seq=5,
            turn=1,
            at_ms=100.0,
            result=result,
            stop_reason="controller_stop",
        )

        output = render_event(event, include_content=False, include_trace=False)

        assert "\n" not in output
        assert output.startswith("{")
        data = json.loads(output)
        assert data["type"] == "search_finished"
        assert "result" in data
        assert "trace" not in data["result"]
        assert "content" not in data["result"]["hits"][0]["hit"]

    def test_other_events_render_as_model_dump_json(self):
        """Non-SearchFinished events render as their model_dump_json()."""
        event = PhaseStarted(
            search_id="s1",
            seq=1,
            turn=1,
            at_ms=0.0,
            phase="plan",
        )

        output = render_event(event, include_content=False, include_trace=False)

        assert output == event.model_dump_json()
