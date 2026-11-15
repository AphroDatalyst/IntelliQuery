"""Object storage for source files, page renders and figure crops."""

import asyncio
import shutil
from pathlib import Path
from typing import Protocol


class ObjectStorage(Protocol):
    async def put(self, key: str, data: bytes) -> None: ...

    async def get(self, key: str) -> bytes: ...

    async def delete_prefix(self, prefix: str) -> None: ...


class LocalStorage:
    """Stores objects as files under a root directory."""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    def _path(self, key: str) -> Path:
        path = (self._root / key).resolve()
        # Keys are built from identifiers we generate, but never trust that blindly.
        if not path.is_relative_to(self._root):
            raise ValueError(f"storage key escapes the storage root: {key!r}")
        return path

    def _put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so readers never observe a partial object.
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)

    def _delete_prefix(self, prefix: str) -> None:
        path = self._path(prefix)
        if path == self._root:
            raise ValueError("refusing to delete the storage root")
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    async def put(self, key: str, data: bytes) -> None:
        await asyncio.to_thread(self._put, key, data)

    async def get(self, key: str) -> bytes:
        return await asyncio.to_thread(self._path(key).read_bytes)

    async def delete_prefix(self, prefix: str) -> None:
        await asyncio.to_thread(self._delete_prefix, prefix)
