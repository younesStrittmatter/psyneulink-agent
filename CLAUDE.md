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

## Stack (planned)

- `uv` for deps and venvs.
- Anthropic Python SDK (with prompt caching, the memory tool, structured 
  outputs) for LLM calls — no rolling-our-own.
- `mcp` Python client for talking to `psyneulink-mcp`.
- `pytest` for tests.

## Status

Scaffold only. Real implementation lands in a separate plan.
