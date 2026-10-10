#!/usr/bin/env python3
"""Run manually in YOUR terminal. Writes a new private .env; never prints a key.

Standard library only. No network request. Refuses to overwrite an existing file.
Codex should not run the interactive prompt through a chat-visible terminal.
"""
from __future__ import annotations

import getpass
import os
from pathlib import Path
import sys


def write_local_env(path: Path, key: str, allow_paid: bool = False) -> None:
    key = key.strip()
    if not key or len(key) < 12 or any(c.isspace() for c in key):
        raise ValueError("A nonempty single-line API key is required; no value was written.")
    if any(c in key for c in ['"', "'", "#", "\\", "\x00"]):
        raise ValueError("Unexpected key characters; no value was written.")
    content = (
        f"OPENROUTER_API_KEY={key}\n"
        "OPENROUTER_MODEL=qwen/qwen3-8b\n"
        "OPENROUTER_PROVIDER=\n"
        f"ALLOW_PAID_RUNS={int(allow_paid)}\n"
        "MAX_SPEND_CAD=5.00\n"
        "MAX_SPEND_USD=3.00\n"
        "CAD_PER_USD=1.50\n"
        "# FX factor is a planning assumption, not a verified exchange rate.\n"
        "LANGSMITH_TRACING=false\n"
        "LANGCHAIN_TRACING_V2=false\n"
        "TOKENIZERS_PARALLELISM=false\n"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    # Exclusive creation refuses an existing .env or a symlink. Mode is user-only.
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    target = Path(__file__).resolve().parent / ".env"
    if target.exists() or target.is_symlink():
        print("A .env already exists. It was not read or changed. Edit it privately in your editor.")
        return 1
    if not sys.stdin.isatty():
        print("Run this helper manually in your own interactive terminal; do not pipe a key into chat.")
        return 1
    print("Local OpenRouter setup. Input is hidden; this script makes no network calls.")
    print("Code must enforce trial spending guards: CAD 5.00 AND USD 3.00 ceilings.")
    print("CAD conversion uses an explicit planning factor of 1.50, not a live exchange rate.")
    try:
        key = getpass.getpass("OpenRouter API key (hidden): ")
        allow = input("Authorize the bounded trial's paid API calls? [y/N]: ").strip().lower() == "y"
        write_local_env(target, key, allow)
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled. No setup completed.")
        return 1
    except (ValueError, FileExistsError):
        print("Setup rejected an invalid key or an existing file. No existing file was overwritten.")
        return 1
    print("Saved local .env with user-only permissions. The key was not displayed.")
    print("Paid-run permission:", "enabled within stated guards" if allow else "disabled")
    print("Review limits privately if needed, then let Codex resume the task.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
