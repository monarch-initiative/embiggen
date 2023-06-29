"""Utilities to skip tests whose files were not changed since last test."""
import os
from typing import List, Union, Optional

import compress_json
from dict_hash import sha256


def cache_or_store(paths: Union[str, List[str]], salt: Optional[str] = None) -> bool:
    """Return whether the given paths were already cached.

    Parameters
    -----------------------
    paths: Union[str, List[str]],
        The paths to check.
    salt: Optional[str] = None,
        The salt to use for the hash.
    """
    if isinstance(paths, str):
        paths = [paths]

    # We check that all paths exist.
    for path in paths:
        if not os.path.exists(path):
            raise ValueError(f"The path {path} does not exists.")

    # We compute the hash of the content of the files.
    total_hash = sha256(
        dict(
            **{path: open(path, "r", encoding="utf8").read() for path in paths},
            salt=salt,
        )
    )

    # We check whether the hash is already stored in the local cache directory.
    exists = os.path.exists(f"tests/cache/{total_hash}.json")

    if not exists:
        metadata = dict(
            total_hash=total_hash,
        )
        # We store the hash in the local cache directory.
        compress_json.dump(metadata, f"tests/cache/{total_hash}.json")

    return exists
