import pytest

from agentic_search.backends.files import FilesBackend
from agentic_search.core.secrets import clear_secrets
from agentic_search.core.types import Document, TextPart
from agentic_search.embedders.local import HashEmbedder

_ROWS = [
    ("d1", "Aspirin reduces fever and relieves headache pain", {"type": "drug", "year": 2020}),
    ("d2", "Ibuprofen is an anti-inflammatory used for pain", {"type": "drug", "year": 2021}),
    ("d3", "The history of the printing press in Europe", {"type": "history", "year": 1999}),
    ("d4", "Acetaminophen treats headache and fever", {"type": "drug", "year": 2019}),
    ("d5", "Medieval castles and their architecture", {"type": "history", "year": 2005}),
]


@pytest.fixture(autouse=True)
def _clear_secrets():
    """Registered secrets are process-global; isolate every test."""
    clear_secrets()
    yield
    clear_secrets()


@pytest.fixture
def medical_docs() -> list[Document]:
    return [Document(doc_id=i, content=[TextPart(text=t)], metadata=m) for i, t, m in _ROWS]


@pytest.fixture
def docs_backend(medical_docs) -> FilesBackend:
    return FilesBackend.from_documents("docs", medical_docs, embedder=HashEmbedder())
