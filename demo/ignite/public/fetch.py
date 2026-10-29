"""Download the public eval datasets into a git-ignored cache, pinned by SHA-256.

A file whose hash does not match is an error, never a silent re-download: the upstream file
changed or a download was cut short, and either way the numbers would not be comparable."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path
from typing import Callable

CACHE_DIR = Path(__file__).parent / ".cache"

_WANDS = "https://raw.githubusercontent.com/wayfair/WANDS/3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5/dataset"

FILES: dict[str, dict[str, tuple[str, str]]] = {
    # MuSiQue v1.0 (CC BY 4.0). The Hugging Face mirror is byte-identical to the official
    # Google Drive zip (checked 2026-10-29), and a plain HTTPS download.
    "musique": {
        "musique_ans_v1.0_dev.jsonl": (
            "https://huggingface.co/datasets/bdsaglam/musique/resolve/main/"
            "musique_ans_v1.0_dev.jsonl",
            "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b"),
    },
    # WANDS (MIT), pinned to commit 3b74dcf (2022-01-18).
    "wands": {
        "product.csv": (f"{_WANDS}/product.csv",
                        "d993926254572e6eba96c8fd87cc549a17fb91ad3748308036eee4cf92b10ac6"),
        "query.csv": (f"{_WANDS}/query.csv",
                      "63b61660560fecc33ec490804c7e2b81402ee3e7c31a9cbb5e03736639f68e95"),
        "label.csv": (f"{_WANDS}/label.csv",
                      "c11fe81ad62f17f56f316b0ec9630ebe8fbe1393578cb0ca4f05c17253a180ef"),
    },
}


class FetchError(RuntimeError):
    pass


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def fetch(dataset: str, cache_dir: Path = CACHE_DIR,
          download: Callable[[str, Path], None] = _download) -> Path:
    """The dataset's folder, with every pinned file present and verified."""
    if dataset not in FILES:
        raise FetchError(f"unknown dataset {dataset!r}; one of {sorted(FILES)}")
    folder = Path(cache_dir) / dataset
    folder.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in FILES[dataset].items():
        path = folder / name
        if not path.exists():
            download(url, path)
        actual = sha256_of(path)
        if actual != expected:
            raise FetchError(f"{dataset}/{name}: SHA-256 is {actual}, expected {expected}. "
                             f"Delete {path} and fetch again, or check whether upstream changed.")
    return folder
