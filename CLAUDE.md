# psyneulink-agent

The modeling agent in the `psyneulink-ai` stack. **Layer 2.** This is the
only repo where modeling logic lives.

## Working with Claude on this project

I'm using this project to learn how to use Claude (Code, API, SDK, MCP) 
efficiently and in a modern way. While we work:

- Surface better tools and idioms.
- Suggest, don't silently do.
- Flag anti-patterns.
- Be concrete.

(Mirrors the rule in `psyneulink-mcp/CLAUDE.md`.)

## Architecture (three repos)

- **`psyneulink-mcp`:** the MCP server wrapping PsyNeuLink. This agent 
  talks to it over stdio MCP.
- **`psyneulink-corpus`:** community brainlike YAMLs + tool feedback 
  Issues. The agent fetches these via the MCP, never directly.
- **`psyneulink-agent` (this repo):** decides *how* to combine personal 
  vs community brainlike views during modeling. Owns prompts, conversation 
  state, modeling strategies.

## Separation of concerns is pure (hard rule)

- No PsyNeuLink imports here. Everything goes through the MCP.
- No writes to `psyneulink-corpus`. Contributions are PRs/Issues filed by 
  humans, not by the agent.
- If the agent needs new MCP tools, they're added in `psyneulink-mcp` 
  first, regenerated, and only then consumed here.

## Multi-repo dev sessions: switch workspace first

A *multi-repo dev session* authors changes in more than one of the
three sibling repos in one sitting (e.g., spawning subagents that
operate in `../psyneulink-mcp/` or `../psyneulink-corpus/`, or
coordinating a label rename that has to land in two repos together).

If you find you need one, **stop and ask the user to open a new Cursor
chat at the parent folder**:

```
~/Documents/code/AutoGrad/psyneulink-ai/
```

That folder has its own `AGENTS.md` and is the correct workspace for
multi-repo dev sessions. The shell sandbox restricts writes to the
workspace root; running cross-repo writes from this sub-repo workspace
forces a permission prompt for every shell call into a sibling. Don't
work around it with `required_permissions: ["all"]` — switch workspaces
once, work freely thereafter.

This is *not* the same as the forbidden cross-repo coupling above. A
multi-repo dev session produces independent commits in independent
repos that each respect the boundary. **Smell test:** if the work
would survive being done in two separate chats on different days with
no shared state, it's a dev-session convenience. If it requires
runtime/import coupling between repos, the polyrepo rule applies and
the design is wrong — fix the design.

## Stack (planned)

- `uv` for deps and venvs.
- Anthropic Python SDK (with prompt caching, the memory tool, structured 
  outputs) for LLM calls — no rolling-our-own.
- `mcp` Python client for talking to `psyneulink-mcp`.
- `pytest` for tests.

## Status

Scaffold only. Real implementation lands in a separate plan.
