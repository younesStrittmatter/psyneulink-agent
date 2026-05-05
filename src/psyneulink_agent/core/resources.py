"""User-attached resources made visible to the LLM during a session.

A :class:`Resource` is anything the user wants the modeling agent to
"have in front of it" — a paper PDF, a behavioural data file, a
previously-saved ``.py`` model. Resources do NOT carry tool semantics;
they're context. The LLM decides whether (and how) to invoke MCP tools
in response to seeing a resource.

Why this layer exists at all (and not just "paste the file path into
the chat"):

* PDFs need to ride as Anthropic ``document`` content blocks (native
  PDF understanding, no OCR or text extraction).
* Data files are far too big to live in the LLM's context — we just
  point at the path and remind the LLM that ``load_psyche_data`` is the
  right MCP tool to actually load them.
* Model files (``.py``) preview their first ~30 lines so the LLM can
  decide whether to ``load_python_script`` them in this session, save
  a renamed copy, etc.

The same ``Resource`` instances flow through every front-end (the
``--chat-sdk`` REPL today, the web UI and ``--run`` mode tomorrow). The
front-end's only job is to translate user gestures (slash commands,
button clicks, YAML fields) into ``session.attach(Resource(...))``.
"""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Resource(ABC):
    """A user-attached resource available to the LLM during the session."""

    @abstractmethod
    def kind(self) -> str:
        """Short tag describing the resource type (e.g. ``"pdf"``)."""

    @abstractmethod
    def label(self) -> str:
        """Short human-readable label (typically the file's basename)."""

    @abstractmethod
    def as_anthropic_blocks(self) -> list[dict[str, Any]]:
        """Return Anthropic Messages content blocks representing this resource.

        Called once on the first user turn that includes this resource.
        Resources that are too large to embed (data files) return a
        text-block describing where the file lives and which MCP tool
        the LLM should reach for to actually load it.
        """

    def summary_line(self) -> str:
        """One-line summary used in slash-command listings and the system prompt."""
        return f"- [{self.kind()}] {self.label()}"


def _resolve_existing(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    return p


class PdfResource(Resource):
    """A PDF reference document attached for the LLM to read natively."""

    def __init__(self, path: str | Path):
        self.path = _resolve_existing(path)
        if self.path.suffix.lower() != ".pdf":
            raise ValueError(f"not a .pdf: {self.path}")

    def kind(self) -> str:
        return "pdf"

    def label(self) -> str:
        return self.path.name

    def as_anthropic_blocks(self) -> list[dict[str, Any]]:
        data = base64.b64encode(self.path.read_bytes()).decode("ascii")
        return [
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": data,
                },
                "title": self.path.name,
            }
        ]


class DataResource(Resource):
    """A behavioural data file (CSV / Parquet / JSONL).

    We don't load the file into the LLM's context — that's what the
    ``load_psyche_data`` MCP tool is for. We just inform the LLM that
    the file exists and where to find it.
    """

    def __init__(self, path: str | Path):
        self.path = _resolve_existing(path)

    def kind(self) -> str:
        return "data"

    def label(self) -> str:
        return self.path.name

    def as_anthropic_blocks(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "text",
                "text": (
                    f"Behavioural data file available at: {self.path}\n"
                    f"Use the load_psyche_data MCP tool with this path "
                    f"when the user asks you to load or fit data."
                ),
            }
        ]


class ModelFileResource(Resource):
    """A previously-saved ``.py`` PsyNeuLink model file.

    We don't import or execute it ourselves — the LLM should choose to
    call ``load_python_script`` on it via the MCP if it wants the model
    re-materialised in the current session.
    """

    PREVIEW_LINES = 30

    def __init__(self, path: str | Path):
        self.path = _resolve_existing(path)
        if self.path.suffix.lower() != ".py":
            raise ValueError(f"not a .py: {self.path}")

    def kind(self) -> str:
        return "model"

    def label(self) -> str:
        return self.path.name

    def as_anthropic_blocks(self) -> list[dict[str, Any]]:
        head = self.path.read_text(encoding="utf-8").splitlines()[: self.PREVIEW_LINES]
        return [
            {
                "type": "text",
                "text": (
                    f"Previously-saved PsyNeuLink model at: {self.path}\n"
                    f"First lines:\n" + "\n".join(head) + "\n"
                    "Use the load_python_script MCP tool to re-materialise it."
                ),
            }
        ]
