import hashlib

import pytest

from demo.ignite.public import fetch as f


def _files(tmp_path, content=b"hello"):
    digest = hashlib.sha256(content).hexdigest()
    return {"toy": {"a.txt": ("https://example.invalid/a.txt", digest)}}, digest


def test_cached_file_with_right_hash_is_not_downloaded(tmp_path, monkeypatch):
    files, _ = _files(tmp_path)
    monkeypatch.setattr(f, "FILES", files)
    (tmp_path / "toy").mkdir()
    (tmp_path / "toy" / "a.txt").write_bytes(b"hello")
    calls = []
    assert f.fetch("toy", tmp_path, download=lambda url, dest: calls.append(url)) == tmp_path / "toy"
    assert calls == []


def test_missing_file_is_downloaded_and_checked(tmp_path, monkeypatch):
    files, _ = _files(tmp_path)
    monkeypatch.setattr(f, "FILES", files)
    calls = []

    def download(url, dest):
        calls.append(url)
        dest.write_bytes(b"hello")

    folder = f.fetch("toy", tmp_path, download=download)
    assert calls == ["https://example.invalid/a.txt"]
    assert (folder / "a.txt").read_bytes() == b"hello"


def test_hash_mismatch_fails(tmp_path, monkeypatch):
    files, digest = _files(tmp_path)
    monkeypatch.setattr(f, "FILES", files)
    with pytest.raises(f.FetchError) as err:
        f.fetch("toy", tmp_path, download=lambda url, dest: dest.write_bytes(b"changed"))
    msg = str(err.value)
    assert "a.txt" in msg and digest in msg and hashlib.sha256(b"changed").hexdigest() in msg


def test_unknown_dataset_fails(tmp_path):
    with pytest.raises(f.FetchError):
        f.fetch("nope", tmp_path)


def test_real_pins_cover_both_datasets():
    assert set(f.FILES) == {"musique", "wands"}
    assert set(f.FILES["wands"]) == {"product.csv", "query.csv", "label.csv"}
    assert all(len(sha) == 64 for files in f.FILES.values() for _, sha in files.values())
