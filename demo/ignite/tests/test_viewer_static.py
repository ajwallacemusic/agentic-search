import re
import subprocess
from pathlib import Path

TESTS = Path(__file__).parent
VIEWER = TESTS.parent / "viewer" / "index.html"


def test_no_external_urls():
    assert not re.search(r"https?://(?!127\.0\.0\.1|localhost)", VIEWER.read_text())


def test_body_font_is_at_least_18px():
    assert re.search(r"body\s*{[^}]*font-size:\s*(1[89]|[2-9]\d)px", VIEWER.read_text())


def test_core_tests_pass():
    subprocess.run(["node", "--test", str(TESTS / "viewer.test.mjs")], check=True)
