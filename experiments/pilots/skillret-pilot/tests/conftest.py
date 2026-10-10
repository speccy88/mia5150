"""Read immutable reference evidence without installing historical runtime state."""

import json
from pathlib import Path
import zipfile

import pytest


@pytest.fixture
def reference_json():
    archive = (
        Path(__file__).resolve().parents[1]
        / "reference_run/MIA5150_skillret_handoff.zip"
    )

    def read(member, jsonl=False):
        # Read only explicitly named archive members; never extract or execute them.
        with zipfile.ZipFile(archive) as snapshot:
            if jsonl and member not in snapshot.namelist():
                return []  # A fully blocked stage may never create its JSONL file.
            text = snapshot.read(member).decode("utf-8")
        if jsonl:
            return [json.loads(line) for line in text.split("\n") if line.strip()]
        return json.loads(text)

    return read
