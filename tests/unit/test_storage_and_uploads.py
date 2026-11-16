import sys

import pytest

from intelliquery.ingestion.service import UploadRejected, validate_upload
from intelliquery.storage import LocalStorage

PDF = b"%PDF-1.7 content"


async def test_storage_round_trip_and_prefix_delete(storage: LocalStorage) -> None:
    await storage.put("documents/a/pages/1.png", b"one")
    await storage.put("documents/a/source.pdf", b"src")
    await storage.put("documents/b/source.pdf", b"other")

    assert await storage.get("documents/a/pages/1.png") == b"one"

    await storage.delete_prefix("documents/a")
    with pytest.raises(FileNotFoundError):
        await storage.get("documents/a/source.pdf")
    assert await storage.get("documents/b/source.pdf") == b"other"


async def test_storage_overwrites_existing_objects(storage: LocalStorage) -> None:
    await storage.put("k", b"old")
    await storage.put("k", b"new")
    assert await storage.get("k") == b"new"


@pytest.mark.parametrize(
    "key",
    [
        "../escape",
        "a/../../escape",
        "/etc/passwd",
        # A drive letter is an absolute path only on Windows; elsewhere it is an
        # ordinary relative name that stays inside the root.
        pytest.param(
            "C:/Windows/x",
            marks=pytest.mark.skipif(sys.platform != "win32", reason="Windows drive path"),
        ),
    ],
)
async def test_storage_rejects_keys_outside_the_root(storage: LocalStorage, key: str) -> None:
    with pytest.raises(ValueError, match="escapes"):
        await storage.put(key, b"x")


async def test_storage_refuses_to_delete_its_root(storage: LocalStorage) -> None:
    with pytest.raises(ValueError, match="root"):
        await storage.delete_prefix("")


def test_upload_must_be_a_pdf_by_content_not_by_name() -> None:
    with pytest.raises(UploadRejected, match="Only PDF"):
        validate_upload("report.pdf", b"MZ\x90 this is an executable", max_bytes=1000)


def test_upload_size_and_emptiness_are_checked() -> None:
    with pytest.raises(UploadRejected, match="empty"):
        validate_upload("a.pdf", b"", max_bytes=1000)
    with pytest.raises(UploadRejected, match="larger than"):
        validate_upload("a.pdf", PDF * 100, max_bytes=1024 * 1024 // 1024)


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("manual.pdf", "manual.pdf"),
        ("../../etc/passwd", "passwd"),
        ("C:\\Users\\me\\manual.pdf", "manual.pdf"),
        ("", "document.pdf"),
    ],
)
def test_upload_filename_is_reduced_to_a_safe_name(given: str, expected: str) -> None:
    assert validate_upload(given, PDF, max_bytes=1000) == expected
