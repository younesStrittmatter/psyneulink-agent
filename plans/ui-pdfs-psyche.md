# Plan: Agent core + front-ends + PDFs + behavioral data (PSYCHE)

**Status:** queued, not yet implemented. Created 2026-05-04.
**Spans repos:** this one (`psyneulink-agent`), `psyneulink-mcp`, plus
two **new** sibling repos that this plan creates. See AGENTS.md at
the parent folder before opening a multi-repo dev session.

## Why

Three independent gaps surfaced in one breath, plus an architectural
constraint they all imply:

1. **Terminal is the wrong UI for modeling.** Once a model has more
   than two nodes, you can't see what the agent built without asking
   for a `repr` dump. Modeling is visual; the chat pane should sit
   next to a live graph of the composition.
2. **No literature in context.** Most modeling sessions start from a
   paper. The agent should read it directly, not be paraphrased to
   by the user.
3. **No data in context either.** A model that can't be evaluated
   against real behavior is a toy. The user already has a defined
   format for behavioral data (PSYCHE, see below) — the agent should
   be able to load it, run the composition over it, fit / optimise
   parameters, and report the result.
4. **Architectural constraint (this is the load-bearing one):**
   PDFs, behavioural data, model fitting, and `.py` save/load are
   **not UI features**. They're core agent capabilities that every
   front-end (interactive terminal, web UI, headless batch) gets
   the same way. The UI is one of multiple front-ends, not the
   privileged home for any of these.

Each capability ships independently and any front-end can use any
combination.

## Layered architecture (this is the contract everything else must respect)

```
                ┌──────────────────────────────────────┐
                │   front-ends (interchangeable)       │
                │                                      │
   terminal ┐   │  --chat (interactive REPL)           │
            │   │  --ui   (web app, psyneulink-ui)     │
   browser ─┤   │  --run  (headless batch / scripted)  │
            │   │                                      │
   cron job ┘   └─────────────────┬────────────────────┘
                                  │
                                  ▼
                ┌──────────────────────────────────────┐
                │   AGENT CORE (this repo's library)   │
                │                                      │
                │  - LLM loop (Anthropic SDK)          │
                │  - PDF attachment                    │
                │  - Session resource registry         │
                │    (PDFs, data files, model files)   │
                │  - Tool routing → MCP                │
                └─────────────────┬────────────────────┘
                                  │  stdio MCP
                                  ▼
                ┌──────────────────────────────────────┐
                │   psyneulink-mcp (tools)             │
                │                                      │
                │  - Generated PNL constructors        │
                │  - Curated composition tools         │
                │  - Curated persistence tools         │
                │    (export_python_script,            │
                │     load_python_script,              │
                │     load_mdf_model, dump_mdf_model)  │
                │  - Curated PSYCHE tools              │
                │    (load_psyche_data,                │
                │     run_composition_on_psyche,       │
                │     fit_composition_to_psyche)       │
                │  - Handles registry                  │
                └─────────────────┬────────────────────┘
                                  │  imports as libraries
                                  ▼
                  ┌────────────────────────────────┐
                  │  psyneulink, psyneulink-psyche │
                  │  (pure data / domain libs)     │
                  └────────────────────────────────┘
```

**Hard rules** (corollaries of `AGENTS.md`):

* PDFs, data files, and `.py` model files are agent-core concerns.
  Front-ends only translate user gestures into agent-core API calls.
* The UI repo (`psyneulink-ui`) imports the agent core as a Python
  library, not the other way around. Removing the UI must not break
  the CLI or the headless mode.
* Capabilities like "fit a model to behavioural data" are MCP tools.
  Front-ends invoke them through the agent's LLM loop. The UI does
  not have its own pathway into the data layer.
* `psyneulink-psyche` is pure data — no PNL, no MCP, no agent. The
  MCP imports it as a library to implement the PSYCHE tools.

## Scope (MVP)

### 1. Agent core refactor — `psyneulink-agent/` (this repo)

Lift the modeling loop out of "spawn `claude` subprocess" into a
proper Python library so any front-end can drive it.

* New module `psyneulink_agent/core/`:
  * `loop.py` — Anthropic SDK tool-calling loop (replaces
    `chat.py`'s subprocess-Claude shortcut). Streams assistant
    chunks and tool calls so a UI can render them live.
  * `session.py` — `Session` object: holds the MCP subprocess,
    conversation history, attached resources (PDFs, data files,
    `.py` model files), and the system prompt. One per
    front-end-instantiated session.
  * `resources.py` — `Resource` types (`PdfResource`,
    `DataResource`, `ModelFileResource`) with `attach(session)` /
    `detach(session)` / `as_anthropic_content_blocks()`. Front-ends
    add resources via this API; the loop attaches them on the
    appropriate turns.
  * `system_prompt.py` — central modeling prompt (single source of
    truth, currently duplicated in `chat.py`).
* Existing `psyneulink-agent --chat` (terminal mode): slimmed down
  to a thin wrapper that constructs a `Session`, runs the loop in
  REPL mode, and accepts new slash-commands `/load-pdf <path>`,
  `/load-data <path>`, `/load-model <path>`, `/save-model [path]`.
* New `psyneulink-agent --run <spec.yaml>` (headless / batch mode):
  read a YAML spec describing goal + resources + acceptance
  criteria, run the loop autonomously, write artifacts (`.py`
  model file, MDF dump, fit report), exit. Closes the loop for
  "run me as a real agent" use cases (cron jobs, CI, sweeps).

### 2. UI — `psyneulink-ui/` (new sibling repo)

Imports the agent core as a library. Adds a web shell on top.

Two-pane web app:

* **Chat pane** (left): textbox + scrollback, streams assistant
  responses as they arrive. Shows tool calls inline (collapsible).
* **Graph pane** (right): live PNG of `composition.show_graph()` for
  the most recently mutated composition. Re-renders whenever a
  composition-mutating tool is called. A dropdown switches between
  compositions when the session has more than one.
* **Resource dock** (bottom or sidebar): upload widget for PDFs,
  CSV/Parquet data files, **and `.py` model files**; list of
  currently-attached resources; delete button per item; "Save
  current model as .py" button. All four affordances translate
  one-to-one into agent-core resource API calls — the UI is
  glue, not logic.

Backend:

* FastAPI process that owns the agent core (`Session`), pushes
  streaming + graph-update events to the frontend via SSE,
  receives uploads + button clicks and forwards them to the
  agent's resource API.

Frontend:

* Whatever ships fastest. React+Vite, htmx+Jinja, or
  Gradio/Streamlit if either supports the SSE + image-pane combo
  cleanly. Pick the smallest stack that doesn't lock us out of
  custom UI later.

Localhost only. Single user. No auth.

### 3. PDF context — agent core

* Anthropic native PDF support: attach uploaded PDFs as
  `{"type": "document", "source": {"type": "base64", ...}}` content
  blocks on the first user turn of the session (and re-attach on
  resume). No text extraction — Claude reads PDFs natively, so
  figures and layout are preserved.
* Lives in `core/resources.py` as `PdfResource`. Both `--chat`
  (`/load-pdf`) and the UI dock and `--run` (YAML spec) attach
  PDFs the same way.
* Multi-PDF: each upload is one document block.
* Soft cap: 5 PDFs / 30 MB total. Warn the user beyond that;
  refuse beyond a hard cap.
* The MCP never sees PDFs — they're context for the LLM, not
  inputs to a tool.

### 4. `.py` model files — agent core (uses MCP tools)

Symmetric load/save, available to every front-end:

* **Save**: agent calls `export_python_script` (MCP tool from
  `psyneulink-mcp/plans/mdf-loader.md`). Available as:
  - `/save-model [path]` slash command in `--chat`,
  - "Save current model as .py" button in the UI,
  - explicit step in a `--run` YAML spec, or
  - implicit autosave at session end (configurable).
* **Load**: agent calls `load_python_script` (also from the MDF
  plan). Available as `/load-model <path>` in `--chat`, file-pick
  in the UI dock, or `model_file:` field in the `--run` spec.
* `ModelFileResource` in `core/resources.py` tracks attached `.py`
  files so the agent can offer to re-save them, diff them, etc.

### 5. Behavioral data — `psyneulink-psyche/` (new sibling repo) + new MCP tools

`psyneulink-psyche/` (pure-Python, no PNL dep):

* Defines `Convention`, `ConventionColumn`,
  `ConventionColumnCategorical`, `ConventionColumnNumeric`,
  `ConventionColumnIndex`, plus the canonical
  `BEHAVIORAL_DATA_CONVENTION` from the user's spec.
* Validators: `validate(df, convention) -> ValidationReport`
  (column-presence, level-membership, uniqueness of
  `(subject_id, trial_global, step)`, …).
* Loaders: `load_csv`, `load_parquet`, `load_jsonl` — each calls
  the validator and returns a typed `BehavioralFrame` (thin wrapper
  around `pandas.DataFrame`).
* Versioned from day one (`PSYCHE_VERSION = "0.1"`); loaders refuse
  mismatched versions with a helpful message.
* Deps: `pandas`, `pydantic` (or `attrs`). No PNL, no MCP, no
  network, no agent.

New MCP tools (in `psyneulink-mcp/src/psyneulink_mcp/tools/curated/psyche.py`,
gated behind `[psyche]` optional extra):

* `load_psyche_data(path) -> {data_handle, schema, n_rows, n_subjects, summary}`
  — load + validate, return a handle for the DataFrame. Extends the
  handles registry to recognise DataFrames (separate prefix
  `df_<id>` so it's obvious in tracebacks what kind of object a
  handle resolves to).
* `describe_psyche_convention() -> {name, version, columns}` — agent
  introspects the schema before suggesting an input mapping.
* `run_composition_on_psyche(composition, data, input_mapping, output_mapping=None) -> {predictions_handle, accuracy?}`
  — for each `subject_id × trial_global × step` row, build the
  composition's `inputs` dict from the mapped columns, run, collect
  output activations. If `output_mapping` is provided, compare to
  `correct_response` / `response` and report row-level + overall
  accuracy. Returns a new handle for a DataFrame that joins
  predictions back to the original rows.
* `fit_composition_to_psyche(composition, data, input_mapping, output_mapping, free_parameters, objective="nll", method="grid"|"de"|"pec") -> {fitted_composition, params, score, report}`
  — wrap PsyNeuLink's `ParameterEstimationComposition` (or a
  simpler grid / scipy.optimize.differential_evolution backend
  when PEC is overkill) to find values of `free_parameters` that
  optimise `objective` against the data. Returns a handle for the
  composition with fitted parameters baked in, the chosen
  parameter values, the achieved score, and a per-row diagnostic
  DataFrame. Available to every front-end through the LLM, just
  like every other MCP tool.

## Repo layout after this work (5 sibling repos)

| Repo | Role | New? |
|------|------|------|
| `psyneulink-mcp` | Tools (incl. new persistence + psyche tools). | existing |
| `psyneulink-corpus` | Brainlikes + issue queue. | existing |
| `psyneulink-agent` | **Agent core** library + three front-ends (`--chat`, `--ui` thin wrapper, `--run`). | existing |
| `psyneulink-ui` | Web frontend + SSE backend. Imports agent core. | **new** |
| `psyneulink-psyche` | Behavioral data convention + loaders. | **new** |

Cross-repo rules from `AGENTS.md` are preserved:

* UI imports the agent core as a Python library (allowed: one
  package depends on another; not "two live services coupled at
  runtime").
* Agent talks to MCP only via stdio MCP.
* MCP imports `psyneulink-psyche` as a library to implement PSYCHE
  tools (allowed for the same reason as above; `psyneulink-psyche`
  is data + validators with no PNL or MCP imports of its own).
* Corpus is data-only, talked to by MCP only.

If any of those arrows ever needs to point the other way, the
design is wrong; fix the design.

## Out of scope for MVP

- Multi-user / hosted deployment. Localhost.
- Editing the model from the graph pane (mouse interactions).
  View-only.
- Animated trial-by-trial dynamics (`composition.show_graph` GIFs).
  Still PNGs after each mutation. Stretch goal.
- PDF figure extraction or OCR. Trust Claude's PDF understanding.
- Saving/sharing chat sessions across machines.
- Promoting PSYCHE to a published BIDS-style standard. Treat it as
  an internal contract for now; let it stabilise before evangelising.
- Cross-subject hierarchical fitting. `fit_composition_to_psyche`
  treats the dataset as a single fit target; per-subject /
  hierarchical fits are a follow-up tool.
- Spec-language for `--run`: keep the YAML schema small (goal text,
  resources list, optional acceptance assertions). A full DSL is
  overkill until users complain.

## Implementation notes

### Agent core (do this first; everything else depends on it)

- Land `core/loop.py`, `core/session.py`, `core/resources.py`,
  `core/system_prompt.py` before touching the UI repo. Do it in
  small commits behind feature flags so the existing `--chat`
  keeps working through the refactor.
- The current `chat.py` (subprocess `claude`) becomes
  `frontends/chat_via_claude_cli.py` and stays as a fallback path
  for users without an Anthropic API key. The new SDK-driven loop
  is the default.
- `Session` is an opaque handle to the front-end; its public API
  is `.send_user_message(text)`, `.attach(resource)`,
  `.detach(resource)`, `.events()` (async iterator yielding
  assistant chunks, tool calls, tool results), `.snapshot()`
  (for autosave). Front-ends never import `anthropic` or `mcp`
  directly.

### UI

- Backend stack: FastAPI + SSE; one `Session` per browser tab.
  System prompt comes from `core/system_prompt.py`, with a UI-
  specific paragraph appended ("you have a graph pane that
  auto-updates after every composition-mutating tool call").
- Graph rendering: `composition.show_graph(output_fmt="pnl")`
  returns a graphviz `Source`; `.render()` to PNG. Cache by
  `(composition_handle, mutation_count)` so unchanged graphs aren't
  re-rendered.
- "Mutation count" is a small new mechanic in the MCP: each
  composition handle gets an integer counter that the curated
  composition tools (`add_node`, `add_linear_pathway`,
  `add_projection`) bump. The UI's backend polls or subscribes via a
  new `mcp__psyneulink__composition_revision(handle)` tool. Cheap.
- Graphviz is a system binary — README documents `brew install
  graphviz` and ships a Dockerfile so users don't have to fight it.

### PDF context

- Pure agent-side. The MCP doesn't see PDFs.
- Anthropic SDK message shape:
  `messages=[{"role": "user", "content": [{"type": "document", ...}, {"type": "text", ...}]}]`.
- Persist uploaded PDFs in a session-scoped temp dir; clean up on
  session end.

### PSYCHE

- The convention object as the user provided it is the canonical
  definition; copy it verbatim into
  `psyneulink_psyche/conventions/behavioral.py` as
  `BEHAVIORAL_DATA_CONVENTION`.
- Make `Convention` and the column classes Pydantic models (or
  attrs-based) so validation errors are structured and the schema
  is JSON-Schema-exportable for non-Python consumers.
- Loaders: small wrappers around `pandas.read_*` that pipe through
  `validate()`. Loaders are the only place data is read; the
  `BehavioralFrame` returned is "validated against version X" by
  construction.
- New repo's `CLAUDE.md` should mirror the existing siblings'
  conventions (separation of concerns, no PNL imports, etc.).

### MCP psyche tools

- Optional extra in `pyproject.toml`:
  `psyche = ["psyneulink-psyche>=0.1", "pandas>=2"]`.
- Module: `src/psyneulink_mcp/tools/curated/psyche.py`. Lazy-import
  both `psyneulink_psyche` and `pandas` at call time; return a
  structured `{"error": "psyche extras not installed; ..."}` on
  ImportError instead of crashing the server.
- `input_mapping` is the load-bearing arg of
  `run_composition_on_psyche`. Document the common patterns in
  the tool description so the agent has a chance: one
  PSYCHE-column-per-input-port, fan-in (multiple columns → one
  port), one-hot encoding for categorical levels, etc.
- `fit_composition_to_psyche` backends:
  - `method="grid"`: dumb cartesian-product sweep, parallelised
    across cores. Cheapest, useful for sanity. 1–3 free params.
  - `method="de"`: `scipy.optimize.differential_evolution`.
    Mid-tier; works for ~10 free params with bounded ranges.
  - `method="pec"`: PsyNeuLink's
    `ParameterEstimationComposition`. Honest answer for the
    "fit a cognitive model to behavioural data" use case;
    integrates with PNL scheduling. Slowest to ship because PEC
    has its own learning curve.
  Ship `grid` and `de` first; `pec` can land in a follow-up.
- DataFrame handles in the registry: extend `handles.py` so that
  the prefix is type-aware (`tm_…` for TransferMechanism, `c_…` for
  Composition, `df_…` for DataFrame, etc.). Optional but pays for
  itself in tracebacks.

## Risks / unknowns

- Anthropic SDK loop maturity in our hands: the `claude` CLI does a
  lot for free (permissions UI, history, slash commands). Replacing
  it with our own loop is more code and we lose the polish.
  Mitigation: keep `psyneulink-agent --chat` (CLI mode) working
  alongside the UI, so users always have a fallback.
- Graphviz install pain on macOS / Windows. Docker image is the
  honest fix.
- PSYCHE convention will evolve — landing the version field on day
  one is non-negotiable.
- "Run a composition over a behavioral DataFrame" can be slow if the
  DataFrame is large and PNL's `Composition.run` isn't compiled.
  Document the perf cliff; suggest `compile=True` in the tool
  description.
- Cost: PDFs in context are billed per token by Anthropic. The soft
  caps + warning are essential.

## Cross-link

`psyneulink-mcp/plans/mdf-loader.md` owns the four MCP tools that
underpin model save / load (`export_python_script`,
`load_python_script`, `load_mdf_model`, `dump_mdf_model`). All four
are reachable identically from every front-end here — there is no
UI-only or chat-only path to model files. Front-ends feature-detect
the tools at session start and hide the corresponding affordances
if missing, so this plan and the MDF plan can land in either order.

## Done when

- The agent core library exists: `Session`, `Resource` types, and
  the SDK-driven loop are importable from `psyneulink_agent.core`
  and have unit tests.
- `psyneulink-agent --chat` (terminal) supports `/load-pdf`,
  `/load-data`, `/load-model`, and `/save-model` — same agent-core
  resource API the UI will use later.
- `psyneulink-agent --run <spec.yaml>` runs to completion on a
  smoke spec ("load this PDF + this CSV, build a model, fit
  parameters, save the resulting `.py`"), exits 0, and writes the
  expected artifacts.
- A user runs `psyneulink-ui` (or whatever the entrypoint becomes),
  opens a browser tab, sees a chat pane + a live-updating graph
  pane + an upload dock + a "Save as .py" button, and can hold a
  modeling conversation. Adding a node in chat causes the graph
  pane to update within ~1s.
- Uploading a PDF and asking "what model does this paper describe?"
  produces a sensible answer in any front-end.
- Uploading a CSV that conforms to `BEHAVIORAL_DATA_CONVENTION` can
  be `load_psyche_data`'d, mapped to a composition via
  `run_composition_on_psyche`, and predictions are returned joined
  to the original rows. `fit_composition_to_psyche` (grid + de
  backends) recovers known-good params on a synthetic dataset.
- `psyneulink-ui/` and `psyneulink-psyche/` exist as siblings of
  `psyneulink-mcp` / `psyneulink-corpus` / `psyneulink-agent`, each
  with their own `CLAUDE.md` reflecting the parent `AGENTS.md`
  conventions, and the parent `AGENTS.md` is updated to list five
  repos instead of three.
- The agent's system prompt mentions every new capability and is
  the SAME prompt regardless of front-end (front-ends append at
  most one paragraph for UI-specific affordances).
- This file is deleted in the same commit that finishes the last of
  the workstreams (per `AGENTS.md`).
