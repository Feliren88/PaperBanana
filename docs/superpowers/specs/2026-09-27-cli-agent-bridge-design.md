# CLI agent bridge: run PaperBanana on Claude Code / Codex instead of API keys

## Goal

Let every PaperBanana agent (retriever, planner, stylist, visualizer, critic,
polish, vanilla) run through a locally logged-in coding-agent CLI — Claude Code
(`claude -p`) or Codex (`codex exec`) — so no provider API key is required.
Selection is by model name, so the CLI (`main.py`), Gradio (`app.py`) and
Streamlit (`demo.py`) front-ends all work unchanged.

Success: with `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`OPENROUTER_API_KEY` all unset, and

```yaml
defaults:
  main_model_name: "claude-code/sonnet"   # or "codex", "codex/gpt-5.5", ...
  image_gen_model_name: "codex"
```

PaperBanana completes a diagram run end to end.

## Verified facts (probed 2026-09-27)

- `claude -p --output-format json --tools ""` answers on subscription OAuth with
  `ANTHROPIC_API_KEY` unset; JSON has `result` and `is_error`.
- `claude --bare` forces `ANTHROPIC_API_KEY` and ignores OAuth — never use it.
- Claude Code's default system prompt costs ~45k tokens per call; passing
  `--system-prompt` replaces it.
- `codex exec` (0.157) supports `-i <image>`, `-o <last-message-file>`,
  `--ephemeral`, `--skip-git-repo-check`, `-s read-only|workspace-write`, `-C dir`,
  and reads the prompt from stdin when given `-`.
- `codex exec` with the stable `image_generation` feature generated a correct
  PNG on subscription auth and saved it to the requested path in its workdir.
- Claude Code has no image generation.

## Design

### Model naming

`claude-code` / `claude-code/<model>` and `codex` / `codex/<model>`. The part
after `/` is passed as `--model` / `-m`; bare names use the CLI's default model.
These prefixes do not collide with existing routing (`claude-` with a hyphen
still goes to the Anthropic API).

### New module `utils/cli_agents.py` (standard library only)

- `parse_cli_model(name) -> (backend, model | None) | None`
- `is_cli_model(name) -> bool`
- `materialize_contents(contents, workdir) -> (prompt_text, image_paths)` —
  joins text parts; decodes both image shapes used in the codebase
  (`{"type":"image","source":{"type":"base64",...}}` and `image_base64`) to
  files in `workdir`.
- `build_command(backend, model, *, system_prompt, image_paths, workdir,
  output_file, generate_image) -> list[str]`
  - Claude: `claude -p --output-format json --no-session-persistence
    --system-prompt <sp> [--model m]`; with images: `--tools Read
    --allowedTools Read --add-dir <workdir>` and the prompt lists the image
    paths to read; without images: `--tools ""`. Image generation → raises
    `CLIAgentUnavailable` (unsupported; not retried).
  - Codex: `codex exec --skip-git-repo-check --ephemeral -C <workdir>
    -o <output_file> [-m m] [-i img ...] -`; sandbox `read-only` for text,
    `workspace-write` for image generation. Codex has no system-prompt flag, so
    the system prompt is prepended to the stdin prompt in a delimited block.
- `call_cli_agent_async(model_name, contents, system_prompt, *, candidate_num,
  max_attempts, retry_delay, timeout) -> list[str]` — text; runs
  `candidate_num` calls concurrently.
- `call_cli_image_generation_async(model_name, contents, system_prompt, *,
  aspect_ratio, max_attempts, retry_delay, timeout) -> list[str]` — returns
  `[base64_png]`; the prompt instructs Codex to generate with its image tool
  and save `output.png` in the workdir; the file is validated as PNG/JPEG
  (magic bytes).
- Subprocesses run via `asyncio.create_subprocess_exec` with prompt on stdin,
  a per-call temporary workdir (removed afterwards), and an environment with
  the corresponding API key variable removed (`ANTHROPIC_API_KEY` for Claude,
  `OPENAI_API_KEY`/`CODEX_API_KEY` for Codex) so the subscription login is
  used. Binaries are resolved via `PATH` (`claude`, `codex`), overridable by
  `PAPERBANANA_CLAUDE_BIN` / `PAPERBANANA_CODEX_BIN`.

### Error handling

Non-zero exit, timeout, `is_error: true`, empty result, or a missing/invalid
image file raise `CLIAgentError` inside an attempt; attempts retry with
`retry_delay`. After `max_attempts` the text call returns `["Error"] *
candidate_num` and the image call returns `["Error"]`, matching the existing
provider functions' contract (agents already treat `"Error"` / empty results).
A missing binary raises immediately with an install hint (no retries).

### Integration points

- `generation_utils.call_model_with_retry_async`: CLI prefixes are checked
  first and dispatched to `call_cli_agent_async` using
  `config.system_instruction` and `config.candidate_count`.
- New `generation_utils.call_cli_image_generation_with_retry_async` thin
  wrapper; `visualizer_agent`, `vanilla_agent`, `polish_agent` check
  `cli_agents.is_cli_model(model)` before their existing image branches
  (polish passes its input image, which Codex receives via `-i`).
- `configs/model_config.template.yaml` and README document the new names.

Out of scope: rewriting `app.py`/`demo.py` refine-image helpers (they call
the Gemini SDK directly), MCP servers, and the earlier host-driven packet
workflow in `docs/no-key-bridge-*.md` (left untouched).

## Testing

`unittest`, matching the repo. `tests/test_cli_agents.py` uses fake `claude` /
`codex` executables (small Python scripts on a temp `PATH`) that record argv,
stdin and environment and emit canned output — this exercises the real
subprocess path without network. Covered: model parsing; content
materialization for both image shapes; exact argv for each backend/mode;
API-key stripping; system prompt handling; JSON parsing; retries then
`"Error"` fallback; timeout; missing binary; image generation file pickup and
validation; Claude image-generation rejection; router dispatch in
`generation_utils`; agent image-branch dispatch. An opt-in live smoke test
(`PAPERBANANA_LIVE_CLI=1`) calls the real CLIs.
