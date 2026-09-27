# CLI Agent Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route PaperBanana's model calls through the logged-in `claude` (Claude Code) or `codex` CLI, selected by model name, so no provider API key is needed.

**Architecture:** A standard-library module `utils/cli_agents.py` turns PaperBanana's generic content lists into one headless CLI invocation per call (prompt on stdin, images as temp files, per-call temp workdir), with retries and the existing `"Error"` fallback contract. `generation_utils.call_model_with_retry_async` dispatches `claude-code[/model]` and `codex[/model]` names to it; the three image-generating agents check `is_cli_model` before their provider branches.

**Tech Stack:** Python 3.13 stdlib (`asyncio` subprocesses, `tempfile`, `shutil`), `unittest`; Claude Code 2.1.x, Codex CLI 0.157.x.

**Spec:** `docs/superpowers/specs/2026-09-27-cli-agent-bridge-design.md`

## Global Constraints

- Model names: `claude-code`, `claude-code/<model>`, `codex`, `codex/<model>`; `claude-…` (hyphen) keeps routing to the Anthropic API.
- Never pass `--bare` to `claude` (it forces `ANTHROPIC_API_KEY`).
- Claude calls always pass `--system-prompt`, `--strict-mcp-config`, `--setting-sources ""`, `--disable-slash-commands`, `--no-session-persistence`, `--output-format json`, `-p`.
- Strip `ANTHROPIC_API_KEY` from Claude's env and `OPENAI_API_KEY`, `CODEX_API_KEY` from Codex's env.
- Binaries: `PAPERBANANA_CLAUDE_BIN` / `PAPERBANANA_CODEX_BIN`, else `claude` / `codex` on `PATH`.
- Failure contract: text → `["Error"] * candidate_num`; image → `["Error"]`; missing binary / Claude image generation → `CLIAgentUnavailable` raised, no retries.
- `utils/cli_agents.py` imports only the standard library.
- Test runner: `.venv/bin/python -m unittest <module>` from the repo root.

## Review Focus

1. Very large prompts (retriever/planner pass many references) must arrive intact — prompt goes via stdin, never argv. Test: 1 MB prompt round-trips (Task 2).
2. Concurrent calls (main.py runs candidates in parallel) must not share files. Test: 5 concurrent Codex image calls return 5 images, and workdirs are cleaned up (Task 2).
3. A CLI that prints non-JSON to stdout (update banners, crashes) must yield `"Error"`, not raise. Test: `garbage` mode (Task 2).
4. Codex may save a JPEG as `output.png`; any valid PNG/JPEG must be accepted, anything else rejected. Test: `jpeg` and `badimage` modes (Task 2).
5. Agent configs without a `system_instruction` must still suppress Claude Code's default 45k-token prompt. Test: router with bare config gets `DEFAULT_SYSTEM_PROMPT` in argv (Tasks 1 & 3).

---

### Task 1: Pure helpers — model parsing, content materialization, prompt and command building

**Files:**
- Create: `utils/cli_agents.py`
- Test: `tests/test_cli_agents.py`

**Interfaces:**
- Produces: `CLIAgentError(RuntimeError)`, `CLIAgentUnavailable(CLIAgentError)`, `DEFAULT_SYSTEM_PROMPT: str`, `IMAGE_OUTPUT_NAME = "output.png"`,
  `parse_cli_model(name) -> tuple[str, str | None] | None`, `is_cli_model(name) -> bool`,
  `resolve_binary(backend: str) -> str`,
  `materialize_contents(contents: list[dict], workdir: Path) -> tuple[str, list[Path]]`,
  `compose_prompt(backend, text, image_paths, system_prompt, *, generate_image=False, aspect_ratio=None) -> str`,
  `build_command(backend, model, *, system_prompt, image_paths, workdir, output_file, generate_image=False) -> list[str]`.

- [ ] **Step 1: Write the failing tests** — `tests/test_cli_agents.py`:

```python
import base64
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from utils import cli_agents
from utils.cli_agents import CLIAgentUnavailable

PNG = b"\x89PNG\r\n\x1a\n" + b"fake-png-body"
JPEG = b"\xff\xd8\xff\xe0" + b"fake-jpeg-body"


class ParseModelTests(unittest.TestCase):
    def test_parses_cli_names(self):
        self.assertEqual(cli_agents.parse_cli_model("claude-code"), ("claude-code", None))
        self.assertEqual(cli_agents.parse_cli_model("claude-code/sonnet"), ("claude-code", "sonnet"))
        self.assertEqual(cli_agents.parse_cli_model("codex"), ("codex", None))
        self.assertEqual(cli_agents.parse_cli_model("codex/gpt-5.5"), ("codex", "gpt-5.5"))

    def test_rejects_api_model_names(self):
        for name in ["claude-sonnet-4-5", "gemini-3.1-pro-preview", "gpt-image-1", "openrouter/codex", "", None]:
            self.assertIsNone(cli_agents.parse_cli_model(name), name)
            self.assertFalse(cli_agents.is_cli_model(name), name)
        self.assertTrue(cli_agents.is_cli_model("codex"))


class MaterializeTests(unittest.TestCase):
    def test_joins_text_and_writes_both_image_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            text, paths = cli_agents.materialize_contents(
                [
                    {"type": "text", "text": "first"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(PNG).decode()}},
                    {"type": "image", "image_base64": base64.b64encode(JPEG).decode()},
                    {"type": "unknown"},
                    {"type": "text", "text": "second"},
                ],
                Path(tmp),
            )
            self.assertEqual(text, "first\n\nsecond")
            self.assertEqual([p.name for p in paths], ["input_1.png", "input_2.jpg"])
            self.assertEqual(paths[0].read_bytes(), PNG)
            self.assertEqual(paths[1].read_bytes(), JPEG)


class ComposePromptTests(unittest.TestCase):
    def test_codex_inlines_system_prompt_claude_does_not(self):
        codex = cli_agents.compose_prompt("codex", "Task.", [], "Be a planner.")
        claude = cli_agents.compose_prompt("claude-code", "Task.", [], "Be a planner.")
        self.assertIn("<system_instructions>\nBe a planner.\n</system_instructions>", codex)
        self.assertNotIn("Be a planner.", claude)
        self.assertIn("Task.", claude)

    def test_claude_lists_image_paths_for_read_tool(self):
        prompt = cli_agents.compose_prompt("claude-code", "Critique.", [Path("/w/input_1.jpg")], "")
        self.assertIn("/w/input_1.jpg", prompt)
        self.assertIn("Read tool", prompt)

    def test_image_generation_prompt_names_output_file_and_ratio(self):
        prompt = cli_agents.compose_prompt("codex", "Draw it.", [], "", generate_image=True, aspect_ratio="16:9")
        self.assertIn(cli_agents.IMAGE_OUTPUT_NAME, prompt)
        self.assertIn("16:9", prompt)
        self.assertIn("image generation tool", prompt)


class BuildCommandTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"PAPERBANANA_CLAUDE_BIN": "/bin/echo", "PAPERBANANA_CODEX_BIN": "/bin/cat"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.workdir = Path("/w")
        self.out = Path("/w/last_message.txt")

    def test_claude_text_command(self):
        cmd = cli_agents.build_command("claude-code", "sonnet", system_prompt="SYS", image_paths=[], workdir=self.workdir, output_file=self.out)
        self.assertEqual(cmd, [
            "/bin/echo", "-p", "--output-format", "json", "--no-session-persistence",
            "--strict-mcp-config", "--setting-sources", "", "--disable-slash-commands",
            "--system-prompt", "SYS", "--tools", "", "--model", "sonnet",
        ])
        self.assertNotIn("--bare", cmd)

    def test_claude_empty_system_prompt_uses_default(self):
        cmd = cli_agents.build_command("claude-code", None, system_prompt="", image_paths=[], workdir=self.workdir, output_file=self.out)
        self.assertEqual(cmd[cmd.index("--system-prompt") + 1], cli_agents.DEFAULT_SYSTEM_PROMPT)
        self.assertNotIn("--model", cmd)

    def test_claude_with_images_enables_read_in_workdir(self):
        cmd = cli_agents.build_command("claude-code", None, system_prompt="S", image_paths=[Path("/w/input_1.jpg")], workdir=self.workdir, output_file=self.out)
        self.assertEqual(cmd[cmd.index("--tools") + 1], "Read")
        self.assertEqual(cmd[cmd.index("--allowedTools") + 1], "Read")
        self.assertEqual(cmd[cmd.index("--add-dir") + 1], "/w")

    def test_claude_cannot_generate_images(self):
        with self.assertRaises(CLIAgentUnavailable):
            cli_agents.build_command("claude-code", None, system_prompt="S", image_paths=[], workdir=self.workdir, output_file=self.out, generate_image=True)

    def test_codex_text_command(self):
        cmd = cli_agents.build_command("codex", "gpt-5.5", system_prompt="S", image_paths=[Path("/w/input_1.jpg")], workdir=self.workdir, output_file=self.out)
        self.assertEqual(cmd, [
            "/bin/cat", "exec", "-i", "/w/input_1.jpg",
            "--skip-git-repo-check", "--ephemeral", "-s", "read-only",
            "-C", "/w", "-o", "/w/last_message.txt", "-m", "gpt-5.5", "-",
        ])

    def test_codex_image_command_can_write_workdir(self):
        cmd = cli_agents.build_command("codex", None, system_prompt="S", image_paths=[], workdir=self.workdir, output_file=self.out, generate_image=True)
        self.assertEqual(cmd[cmd.index("-s") + 1], "workspace-write")
        self.assertEqual(cmd[-1], "-")

    def test_missing_binary_is_unavailable(self):
        with mock.patch.dict(os.environ, {"PAPERBANANA_CLAUDE_BIN": "/nonexistent/claude"}):
            with self.assertRaisesRegex(CLIAgentUnavailable, "claude"):
                cli_agents.resolve_binary("claude-code")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m unittest tests.test_cli_agents -v`
Expected: `ImportError: cannot import name 'cli_agents' from 'utils'` (module missing).

- [ ] **Step 3: Implement** — `utils/cli_agents.py`:

```python
"""Run PaperBanana model calls through a logged-in coding-agent CLI.

Model names ``claude-code[/<model>]`` and ``codex[/<model>]`` route a call to
``claude -p`` or ``codex exec`` instead of a provider API, so the user's
Claude Code / Codex subscription login is used and no API key is needed.
Standard library only.
"""

import base64
import os
import shutil
from pathlib import Path

# backend -> (binary override env var, default binary, API-key env vars to strip)
BACKENDS = {
    "claude-code": ("PAPERBANANA_CLAUDE_BIN", "claude", ("ANTHROPIC_API_KEY",)),
    "codex": ("PAPERBANANA_CODEX_BIN", "codex", ("OPENAI_API_KEY", "CODEX_API_KEY")),
}
IMAGE_OUTPUT_NAME = "output.png"
# Replaces Claude Code's own ~45k-token coding-agent prompt when an agent has none.
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Answer the user's request directly."
_IMAGE_SUFFIXES = {"image/png": ".png", "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/webp": ".webp", "image/gif": ".gif"}


class CLIAgentError(RuntimeError):
    """A single CLI call failed; the caller may retry."""


class CLIAgentUnavailable(CLIAgentError):
    """The requested CLI cannot serve this call at all; retrying will not help."""


def parse_cli_model(name):
    if not isinstance(name, str):
        return None
    backend, _, model = name.partition("/")
    if backend not in BACKENDS:
        return None
    return backend, model or None


def is_cli_model(name):
    return parse_cli_model(name) is not None


def resolve_binary(backend):
    env_var, default, _ = BACKENDS[backend]
    binary = shutil.which(os.environ.get(env_var) or default)
    if not binary:
        raise CLIAgentUnavailable(
            f"'{default}' CLI not found. Install it and log in, or set {env_var} to its path."
        )
    return binary


def materialize_contents(contents, workdir):
    """Split a generic content list into prompt text and image files in ``workdir``."""
    texts, image_paths = [], []
    for item in contents:
        if item.get("type") == "text":
            texts.append(item["text"])
        elif item.get("type") == "image":
            source = item.get("source", {})
            if source.get("type") == "base64":
                data, media_type = source["data"], source.get("media_type", "image/jpeg")
            elif "image_base64" in item:
                data, media_type = item["image_base64"], "image/jpeg"
            else:
                continue
            path = Path(workdir) / f"input_{len(image_paths) + 1}{_IMAGE_SUFFIXES.get(media_type, '.jpg')}"
            path.write_bytes(base64.b64decode(data))
            image_paths.append(path)
    return "\n\n".join(texts), image_paths


def compose_prompt(backend, text, image_paths, system_prompt, *, generate_image=False, aspect_ratio=None):
    parts = []
    if backend == "codex" and system_prompt:
        # codex exec has no system-prompt flag.
        parts.append(f"<system_instructions>\n{system_prompt}\n</system_instructions>")
    if backend == "claude-code" and image_paths:
        parts.append(
            "Input images (open each with the Read tool before answering):\n"
            + "\n".join(str(p) for p in image_paths)
        )
    parts.append(text)
    if generate_image:
        ratio = f" with aspect ratio {aspect_ratio}" if aspect_ratio else ""
        attached = " The attached image(s) are the input to edit." if image_paths else ""
        parts.append(
            f"Use your image generation tool to create this image{ratio}.{attached} "
            f"Save the final image as the PNG file {IMAGE_OUTPUT_NAME} in the current working directory. "
            "Do not create any other files. Reply with only the file path."
        )
    else:
        parts.append("Respond with only the requested output. Do not run commands or modify files.")
    return "\n\n".join(parts)


def build_command(backend, model, *, system_prompt, image_paths, workdir, output_file, generate_image=False):
    if backend == "claude-code":
        if generate_image:
            raise CLIAgentUnavailable(
                "Claude Code cannot generate images; set image_gen_model_name to 'codex' "
                "(or an image API model)."
            )
        cmd = [
            resolve_binary(backend), "-p", "--output-format", "json", "--no-session-persistence",
            "--strict-mcp-config", "--setting-sources", "", "--disable-slash-commands",
            "--system-prompt", system_prompt or DEFAULT_SYSTEM_PROMPT,
        ]
        if image_paths:
            cmd += ["--tools", "Read", "--allowedTools", "Read", "--add-dir", str(workdir)]
        else:
            cmd += ["--tools", ""]
        if model:
            cmd += ["--model", model]
        return cmd

    cmd = [resolve_binary(backend), "exec"]
    for path in image_paths:  # before other flags: -i takes a variable number of values
        cmd += ["-i", str(path)]
    cmd += [
        "--skip-git-repo-check", "--ephemeral",
        "-s", "workspace-write" if generate_image else "read-only",
        "-C", str(workdir), "-o", str(output_file),
    ]
    if model:
        cmd += ["-m", model]
    cmd.append("-")  # prompt on stdin
    return cmd
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m unittest tests.test_cli_agents -v`
Expected: all tests OK.

- [ ] **Step 5: Commit**

```bash
git add utils/cli_agents.py tests/test_cli_agents.py
git commit -m "feat: add CLI agent command builder for Claude Code and Codex"
```

---

### Task 2: Subprocess execution, retries, text and image calls

**Files:**
- Modify: `utils/cli_agents.py` (append)
- Test: `tests/test_cli_agents.py` (append `FakeCLITests`)

**Interfaces:**
- Consumes: everything from Task 1.
- Produces:
  `async call_cli_agent_async(model_name, contents, system_prompt="", *, candidate_num=1, max_attempts=3, retry_delay=5, timeout=DEFAULT_TIMEOUT, error_context="") -> list[str]`,
  `async call_cli_image_generation_async(model_name, contents, system_prompt="", *, aspect_ratio=None, max_attempts=3, retry_delay=5, timeout=DEFAULT_TIMEOUT, error_context="") -> list[str]` (base64 image or `"Error"`),
  `DEFAULT_TIMEOUT = 900`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_cli_agents.py` (before the `__main__` guard) a fake CLI that records argv/stdin/env and behaves per `FAKE_CLI_MODE`:

```python
import asyncio
import json
import stat
import time

FAKE_CLI = r'''#!/usr/bin/env python3
import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
stdin = sys.stdin.read()
record_path = os.environ["FAKE_CLI_RECORD"]
with open(record_path, "a") as f:
    f.write(json.dumps({
        "args": args, "stdin": stdin, "cwd": os.getcwd(),
        "existing": [p for p in args + stdin.split() if os.path.isfile(p)],
        "env": {k: os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "CODEX_API_KEY")},
    }) + "\n")
calls = len(open(record_path).read().splitlines())
mode = os.environ.get("FAKE_CLI_MODE", "ok")
text = os.environ.get("FAKE_CLI_TEXT", "fake answer")
if mode == "fail" or (mode == "fail_once" and calls == 1):
    sys.stderr.write("boom")
    sys.exit(2)
if mode == "sleep":
    time.sleep(5)
if args[0] == "exec":
    workdir = Path(args[args.index("-C") + 1])
    Path(args[args.index("-o") + 1]).write_text(text)
    image = {"image": PNG_BYTES, "jpeg": JPEG_BYTES, "badimage": b"not an image"}.get(mode)
    if image is not None:
        (workdir / "output.png").write_bytes(image)
elif mode == "garbage":
    print("Update available! Run brew upgrade.")
else:
    print(json.dumps({"type": "result", "result": text, "is_error": mode == "is_error"}))
'''.replace("PNG_BYTES", repr(PNG)).replace("JPEG_BYTES", repr(JPEG))

TEXT_CONTENTS = [{"type": "text", "text": "Plan the figure."}]


class FakeCLITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        script = Path(self.tmp.name) / "fake_cli"
        script.write_text(FAKE_CLI)
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        self.record = Path(self.tmp.name) / "record.jsonl"
        patcher = mock.patch.dict(os.environ, {
            "PAPERBANANA_CLAUDE_BIN": str(script),
            "PAPERBANANA_CODEX_BIN": str(script),
            "FAKE_CLI_RECORD": str(self.record),
            "ANTHROPIC_API_KEY": "sk-ant-secret",
            "OPENAI_API_KEY": "sk-openai-secret",
            "CODEX_API_KEY": "codex-secret",
        })
        patcher.start()
        self.addCleanup(patcher.stop)

    def mode(self, mode, text=None):
        env = {"FAKE_CLI_MODE": mode}
        if text is not None:
            env["FAKE_CLI_TEXT"] = text
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)

    def calls(self):
        if not self.record.exists():
            return []
        return [json.loads(line) for line in self.record.read_text().splitlines()]

    def text(self, model, contents=TEXT_CONTENTS, **kw):
        kw.setdefault("retry_delay", 0)
        return asyncio.run(cli_agents.call_cli_agent_async(model, contents, **kw))

    def image(self, model, contents=TEXT_CONTENTS, **kw):
        kw.setdefault("retry_delay", 0)
        return asyncio.run(cli_agents.call_cli_image_generation_async(model, contents, **kw))

    def test_claude_text_uses_stdin_and_strips_its_api_key(self):
        self.mode("ok", "the plan")
        self.assertEqual(self.text("claude-code/sonnet", system_prompt="SYS"), ["the plan"])
        call = self.calls()[0]
        self.assertIn("Plan the figure.", call["stdin"])
        self.assertIsNone(call["env"]["ANTHROPIC_API_KEY"])
        self.assertEqual(call["args"][call["args"].index("--system-prompt") + 1], "SYS")

    def test_candidates_run_as_separate_calls(self):
        self.mode("ok", "x")
        self.assertEqual(self.text("claude-code", candidate_num=3), ["x", "x", "x"])
        self.assertEqual(len(self.calls()), 3)

    def test_codex_text_reads_last_message_and_strips_openai_keys(self):
        self.mode("ok", "codex plan")
        self.assertEqual(self.text("codex", system_prompt="Be a stylist."), ["codex plan"])
        call = self.calls()[0]
        self.assertIn("Be a stylist.", call["stdin"])
        self.assertIsNone(call["env"]["OPENAI_API_KEY"])
        self.assertIsNone(call["env"]["CODEX_API_KEY"])

    def test_claude_reads_image_files_that_are_removed_afterwards(self):
        self.mode("ok", "critique")
        contents = TEXT_CONTENTS + [{"type": "image", "image_base64": base64.b64encode(JPEG).decode()}]
        self.assertEqual(self.text("claude-code", contents), ["critique"])
        call = self.calls()[0]
        self.assertEqual(len(call["existing"]), 1)
        self.assertEqual(call["args"][call["args"].index("--add-dir") + 1], call["cwd"])
        self.assertFalse(Path(call["existing"][0]).exists())

    def test_persistent_failure_returns_error_per_candidate(self):
        self.mode("fail")
        self.assertEqual(self.text("claude-code", candidate_num=2, max_attempts=2), ["Error", "Error"])
        self.assertEqual(len(self.calls()), 4)

    def test_transient_failure_is_retried(self):
        self.mode("fail_once", "second try")
        self.assertEqual(self.text("codex", max_attempts=3), ["second try"])
        self.assertEqual(len(self.calls()), 2)

    def test_is_error_and_garbage_output_become_error(self):
        for mode in ("is_error", "garbage"):
            with self.subTest(mode=mode):
                self.mode(mode)
                self.assertEqual(self.text("claude-code", max_attempts=1), ["Error"])

    def test_timeout_becomes_error(self):
        self.mode("sleep")
        start = time.monotonic()
        self.assertEqual(self.text("claude-code", max_attempts=1, timeout=0.5), ["Error"])
        self.assertLess(time.monotonic() - start, 4)

    def test_missing_binary_raises_without_retry(self):
        with mock.patch.dict(os.environ, {"PAPERBANANA_CODEX_BIN": "/nonexistent/codex"}):
            with self.assertRaises(CLIAgentUnavailable):
                self.text("codex", max_attempts=5)

    def test_large_prompt_arrives_intact(self):
        self.mode("ok")
        big = "r" * 1_000_000
        self.text("claude-code", [{"type": "text", "text": big}])
        self.assertIn(big, self.calls()[0]["stdin"])

    def test_codex_generates_png_and_jpeg(self):
        for mode, payload in (("image", PNG), ("jpeg", JPEG)):
            with self.subTest(mode=mode):
                self.mode(mode)
                result = self.image("codex", system_prompt="Illustrator.", aspect_ratio="16:9")
                self.assertEqual(base64.b64decode(result[0]), payload)
        call = self.calls()[-1]
        self.assertEqual(call["args"][call["args"].index("-s") + 1], "workspace-write")
        self.assertIn("16:9", call["stdin"])

    def test_invalid_or_missing_image_becomes_error(self):
        for mode in ("badimage", "ok"):
            with self.subTest(mode=mode):
                self.mode(mode)
                self.assertEqual(self.image("codex", max_attempts=1), ["Error"])

    def test_image_editing_passes_input_image(self):
        self.mode("image")
        contents = TEXT_CONTENTS + [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(JPEG).decode()}}]
        self.image("codex", contents)
        call = self.calls()[0]
        self.assertEqual(call["args"][1:3], ["exec", "-i"])
        self.assertEqual(len(call["existing"]), 1)

    def test_concurrent_image_calls_are_isolated(self):
        self.mode("image")

        async def run_many():
            return await asyncio.gather(*(
                cli_agents.call_cli_image_generation_async("codex", TEXT_CONTENTS, retry_delay=0) for _ in range(5)
            ))

        results = asyncio.run(run_many())
        self.assertEqual([r[0] for r in results], [base64.b64encode(PNG).decode()] * 5)
        workdirs = {c["cwd"] for c in self.calls()}
        self.assertEqual(len(workdirs), 5)
        self.assertFalse(any(Path(w).exists() for w in workdirs))

    def test_claude_image_generation_is_unavailable(self):
        with self.assertRaises(CLIAgentUnavailable):
            self.image("claude-code")
        self.assertEqual(self.calls(), [])
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m unittest tests.test_cli_agents -v`
Expected: `FakeCLITests` fail with `AttributeError: module 'utils.cli_agents' has no attribute 'call_cli_agent_async'`.

- [ ] **Step 3: Implement** — append to `utils/cli_agents.py` (add `import asyncio`, `import json`, `import tempfile` to the imports):

```python
DEFAULT_TIMEOUT = 900  # seconds; Codex image generation takes ~1 minute


def _subprocess_env(backend):
    stripped = BACKENDS[backend][2]
    return {k: v for k, v in os.environ.items() if k not in stripped}


async def _run(cmd, prompt, *, cwd, env, timeout):
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=cwd, env=env,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(prompt.encode()), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise CLIAgentError(f"timed out after {timeout}s")
    if proc.returncode != 0:
        raise CLIAgentError(f"exited with {proc.returncode}: {stderr.decode(errors='replace')[-800:].strip()}")
    return stdout.decode(errors="replace")


def _parse_text(backend, stdout, output_file):
    if backend == "claude-code":
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            raise CLIAgentError(f"unexpected output: {stdout[:300]!r}")
        if payload.get("is_error"):
            raise CLIAgentError(f"reported an error: {payload.get('result')!r}")
        text = payload.get("result") or ""
    else:
        text = output_file.read_text() if output_file.exists() else ""
    if not text.strip():
        raise CLIAgentError("returned an empty response")
    return text.strip()


def _read_image(path):
    if not path.exists():
        raise CLIAgentError(f"did not produce {path.name}")
    data = path.read_bytes()
    if not (data.startswith(b"\x89PNG\r\n\x1a\n") or data.startswith(b"\xff\xd8\xff")):
        raise CLIAgentError(f"{path.name} is not a PNG or JPEG image")
    return base64.b64encode(data).decode()


async def _call_once(backend, model, contents, system_prompt, timeout, generate_image=False, aspect_ratio=None):
    with tempfile.TemporaryDirectory(prefix="paperbanana-cli-") as tmp:
        workdir = Path(tmp).resolve()
        text, image_paths = materialize_contents(contents, workdir)
        output_file = workdir / "last_message.txt"
        cmd = build_command(
            backend, model, system_prompt=system_prompt, image_paths=image_paths,
            workdir=workdir, output_file=output_file, generate_image=generate_image,
        )
        prompt = compose_prompt(
            backend, text, image_paths, system_prompt,
            generate_image=generate_image, aspect_ratio=aspect_ratio,
        )
        stdout = await _run(cmd, prompt, cwd=workdir, env=_subprocess_env(backend), timeout=timeout)
        if generate_image:
            return _read_image(workdir / IMAGE_OUTPUT_NAME)
        return _parse_text(backend, stdout, output_file)


async def _with_retry(attempt_fn, max_attempts, retry_delay, label):
    for attempt in range(1, max_attempts + 1):
        try:
            return await attempt_fn()
        except CLIAgentUnavailable:
            raise
        except CLIAgentError as e:
            print(f"[CLI agent] {label} attempt {attempt}/{max_attempts} failed: {e}")
            if attempt < max_attempts:
                await asyncio.sleep(retry_delay)
    return None


def _require(model_name):
    parsed = parse_cli_model(model_name)
    if parsed is None:
        raise ValueError(f"{model_name!r} is not a CLI agent model name")
    return parsed


async def call_cli_agent_async(
    model_name, contents, system_prompt="", *, candidate_num=1,
    max_attempts=3, retry_delay=5, timeout=DEFAULT_TIMEOUT, error_context="",
):
    """Text generation through the CLI. Returns ``candidate_num`` strings; ``"Error"`` for failures."""
    backend, model = _require(model_name)
    label = f"{model_name} {error_context}".strip()

    async def one_candidate():
        result = await _with_retry(
            lambda: _call_once(backend, model, contents, system_prompt, timeout),
            max_attempts, retry_delay, label,
        )
        return result if result is not None else "Error"

    return list(await asyncio.gather(*(one_candidate() for _ in range(max(1, candidate_num or 1)))))


async def call_cli_image_generation_async(
    model_name, contents, system_prompt="", *, aspect_ratio=None,
    max_attempts=3, retry_delay=5, timeout=DEFAULT_TIMEOUT, error_context="",
):
    """Image generation through the CLI. Returns ``[base64_image]`` or ``["Error"]``."""
    backend, model = _require(model_name)
    result = await _with_retry(
        lambda: _call_once(
            backend, model, contents, system_prompt, timeout,
            generate_image=True, aspect_ratio=aspect_ratio,
        ),
        max_attempts, retry_delay, f"{model_name} image {error_context}".strip(),
    )
    return [result if result is not None else "Error"]
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m unittest tests.test_cli_agents -v`
Expected: all OK.

- [ ] **Step 5: Commit**

```bash
git add utils/cli_agents.py tests/test_cli_agents.py
git commit -m "feat: run text and image calls through Claude Code / Codex CLIs"
```

---

### Task 3: Wire into the router and the image-generating agents

**Files:**
- Modify: `utils/generation_utils.py` (import; router head; new wrapper)
- Modify: `agents/visualizer_agent.py`, `agents/vanilla_agent.py`, `agents/polish_agent.py` (image branch)
- Test: `tests/test_cli_agent_routing.py`

**Interfaces:**
- Consumes: `cli_agents.is_cli_model`, `cli_agents.call_cli_agent_async`, `cli_agents.call_cli_image_generation_async`.
- Produces: `generation_utils.is_cli_model`, `async generation_utils.call_cli_image_generation_with_retry_async(model_name, contents, config: dict, max_attempts=5, retry_delay=30, error_context="") -> list[str]` where `config` keys are `system_prompt`, `aspect_ratio`.

- [ ] **Step 1: Write the failing tests** — `tests/test_cli_agent_routing.py`:

```python
import asyncio
import base64
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from google.genai import types

from agents.polish_agent import PolishAgent
from agents.vanilla_agent import VanillaAgent
from agents.visualizer_agent import VisualizerAgent
from utils import generation_utils
from utils.config import ExpConfig

CONTENTS = [{"type": "text", "text": "Describe."}]


class RouterTests(unittest.TestCase):
    def test_cli_model_is_dispatched_without_any_api_client(self):
        fake = mock.AsyncMock(return_value=["a", "b"])
        with mock.patch.object(generation_utils, "gemini_client", None), \
             mock.patch.object(generation_utils, "anthropic_client", None), \
             mock.patch.object(generation_utils, "openai_client", None), \
             mock.patch.object(generation_utils, "openrouter_client", None), \
             mock.patch("utils.cli_agents.call_cli_agent_async", fake):
            result = asyncio.run(generation_utils.call_model_with_retry_async(
                "claude-code/sonnet", CONTENTS,
                types.GenerateContentConfig(system_instruction="SYS", candidate_count=2),
                max_attempts=4, retry_delay=1, error_context="planner",
            ))
        self.assertEqual(result, ["a", "b"])
        fake.assert_awaited_once_with(
            "claude-code/sonnet", CONTENTS, system_prompt="SYS", candidate_num=2,
            max_attempts=4, retry_delay=1, error_context="planner",
        )

    def test_config_without_system_instruction_passes_empty_prompt(self):
        fake = mock.AsyncMock(return_value=["a"])
        with mock.patch("utils.cli_agents.call_cli_agent_async", fake):
            asyncio.run(generation_utils.call_model_with_retry_async("codex", CONTENTS, types.GenerateContentConfig()))
        self.assertEqual(fake.call_args.kwargs["system_prompt"], "")
        self.assertEqual(fake.call_args.kwargs["candidate_num"], 1)

    def test_image_wrapper_forwards_prompt_and_ratio(self):
        fake = mock.AsyncMock(return_value=["img"])
        with mock.patch("utils.cli_agents.call_cli_image_generation_async", fake):
            result = asyncio.run(generation_utils.call_cli_image_generation_with_retry_async(
                "codex", CONTENTS, {"system_prompt": "SYS", "aspect_ratio": "16:9"}, max_attempts=2, retry_delay=0,
            ))
        self.assertEqual(result, ["img"])
        fake.assert_awaited_once_with(
            "codex", CONTENTS, system_prompt="SYS", aspect_ratio="16:9",
            max_attempts=2, retry_delay=0, error_context="",
        )


def _config(work_dir, task="diagram"):
    return ExpConfig(
        dataset_name="PaperBananaBench", task_name=task, exp_mode="vanilla", retrieval_setting="none",
        main_model_name="claude-code/sonnet", image_gen_model_name="codex", work_dir=work_dir,
    )


class AgentImageDispatchTests(unittest.TestCase):
    def run_with_cli_image(self, make_agent, data, key):
        async def run():
            with TemporaryDirectory() as tmp:
                agent = make_agent(Path(tmp))
                with mock.patch("utils.generation_utils.call_cli_image_generation_with_retry_async",
                                mock.AsyncMock(return_value=["png"])) as call, \
                     mock.patch("utils.generation_utils.call_gemini_with_retry_async",
                                mock.AsyncMock(side_effect=AssertionError("Gemini must not be called"))), \
                     mock.patch("utils.image_utils.convert_png_b64_to_jpg_b64", mock.Mock(return_value="jpeg")):
                    return await agent.process(data), call
        output, call = asyncio.run(run())
        self.assertEqual(output[key], "jpeg")
        self.assertEqual(call.call_args.kwargs["model_name"], "codex")
        return call

    def test_visualizer_uses_cli_image_generation(self):
        call = self.run_with_cli_image(
            lambda d: VisualizerAgent(exp_config=_config(d)),
            {"candidate_id": "c", "target_diagram_desc0": "Render it.", "additional_info": {"rounded_ratio": "16:9"}},
            "target_diagram_desc0_base64_jpg",
        )
        self.assertEqual(call.call_args.kwargs["config"]["aspect_ratio"], "16:9")

    def test_vanilla_uses_cli_image_generation(self):
        self.run_with_cli_image(
            lambda d: VanillaAgent(exp_config=_config(d)),
            {"content": "Method.", "visual_intent": "Caption.", "additional_info": {"rounded_ratio": "1:1"}},
            "vanilla_diagram_base64_jpg",
        )

    def test_polish_passes_source_image_to_cli(self):
        def make(work_dir):
            (work_dir / "style_guides").mkdir()
            (work_dir / "style_guides" / "neurips2025_diagram_style_guide.md").write_text("Use clean fonts.")
            image_dir = work_dir / "data/PaperBananaBench/diagram"
            image_dir.mkdir(parents=True)
            (image_dir / "gt.jpg").write_bytes(b"\xff\xd8\xffjpeg")
            return PolishAgent(exp_config=_config(work_dir))

        with mock.patch("utils.generation_utils.call_model_with_retry_async",
                        mock.AsyncMock(return_value=["Increase font size."])):
            call = self.run_with_cli_image(make, {"path_to_gt_image": "gt.jpg"}, "polished_diagram_base64_jpg")
        contents = call.call_args.kwargs["contents"]
        self.assertEqual(base64.b64decode(contents[1]["source"]["data"]), b"\xff\xd8\xffjpeg")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m unittest tests.test_cli_agent_routing -v`
Expected: router tests fail (`RuntimeError: No API client available` / missing attribute `call_cli_image_generation_with_retry_async`); agent tests fail with "Gemini must not be called".

- [ ] **Step 3: Implement**

`utils/generation_utils.py` — after `from openai import AsyncOpenAI`:

```python
from utils import cli_agents
from utils.cli_agents import is_cli_model
```

At the top of `call_model_with_retry_async`'s body (before the provider-prefix checks), and update its docstring rule 1 to mention `claude-code`/`codex`:

```python
    # Logged-in coding-agent CLIs (Claude Code / Codex) need no API key.
    if is_cli_model(model_name):
        return await cli_agents.call_cli_agent_async(
            model_name,
            contents,
            system_prompt=getattr(config, "system_instruction", None) or "",
            candidate_num=getattr(config, "candidate_count", None) or 1,
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
        )
```

Before `call_model_with_retry_async`, add:

```python
async def call_cli_image_generation_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """Generate an image through a logged-in Codex CLI. ``config``: system_prompt, aspect_ratio."""
    return await cli_agents.call_cli_image_generation_async(
        model_name,
        contents,
        system_prompt=config.get("system_prompt", ""),
        aspect_ratio=config.get("aspect_ratio"),
        max_attempts=max_attempts,
        retry_delay=retry_delay,
        error_context=error_context,
    )
```

`agents/visualizer_agent.py` — the image branch becomes (`if` added, old `if "gpt-image"` becomes `elif`):

```python
            if cfg["use_image_generation"]:
                if generation_utils.is_cli_model(self.model_name):
                    response_list = await generation_utils.call_cli_image_generation_with_retry_async(
                        model_name=self.model_name,
                        contents=content_list,
                        config={"system_prompt": self.system_prompt, "aspect_ratio": aspect_ratio},
                        max_attempts=5,
                        retry_delay=30,
                    )
                elif "gpt-image" in self.model_name:
```

`agents/vanilla_agent.py` — identical change at its `if "gpt-image" in self.model_name:` (indentation one level shallower).

`agents/polish_agent.py` — in the polish step, before `if generation_utils.openrouter_client is not None:`:

```python
            if generation_utils.is_cli_model(self.image_gen_model_name):
                response_list = await generation_utils.call_cli_image_generation_with_retry_async(
                    model_name=self.image_gen_model_name,
                    contents=content_list,
                    config={"system_prompt": self.system_prompt, "aspect_ratio": aspect_ratio},
                    max_attempts=5,
                    retry_delay=30,
                )
            elif generation_utils.openrouter_client is not None:
```

- [ ] **Step 4: Run to verify pass, plus the full suite**

Run: `.venv/bin/python -m unittest tests.test_cli_agent_routing tests.test_cli_agents tests.test_legacy_generation_options tests.test_legacy_plot_agents tests.test_legacy_ui_result_keys tests.test_planner_metaphor tests.test_plot_execution`
Expected: all OK (31 pre-existing + new).

- [ ] **Step 5: Commit**

```bash
git add utils/generation_utils.py agents/visualizer_agent.py agents/vanilla_agent.py agents/polish_agent.py tests/test_cli_agent_routing.py
git commit -m "feat: route claude-code/codex model names through the CLI bridge"
```

---

### Task 4: Live smoke test, docs, and end-to-end run

**Files:**
- Create: `tests/test_cli_agents_live.py`
- Modify: `configs/model_config.template.yaml`, `README.md` (new subsection near the API-key setup)

- [ ] **Step 1: Write the opt-in live test**

```python
"""Real Claude Code / Codex calls. Run with PAPERBANANA_LIVE_CLI=1 (uses your subscription)."""
import asyncio
import base64
import os
import unittest

from utils import cli_agents

LIVE = os.environ.get("PAPERBANANA_LIVE_CLI") == "1"


@unittest.skipUnless(LIVE, "set PAPERBANANA_LIVE_CLI=1 to call the real CLIs")
class LiveCLITests(unittest.TestCase):
    def test_claude_code_text(self):
        [answer] = asyncio.run(cli_agents.call_cli_agent_async(
            "claude-code/haiku", [{"type": "text", "text": "Reply with exactly: PONG"}], "You are terse.",
        ))
        self.assertIn("PONG", answer)

    def test_codex_text(self):
        [answer] = asyncio.run(cli_agents.call_cli_agent_async(
            "codex", [{"type": "text", "text": "Reply with exactly: PONG"}], "You are terse.",
        ))
        self.assertIn("PONG", answer)

    def test_codex_image(self):
        [image] = asyncio.run(cli_agents.call_cli_image_generation_async(
            "codex", [{"type": "text", "text": "A blue square on a white background."}], aspect_ratio="1:1",
        ))
        self.assertNotEqual(image, "Error")
        self.assertTrue(base64.b64decode(image)[:4] in (b"\x89PNG", b"\xff\xd8\xff\xe0", b"\xff\xd8\xff\xe1"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it live**

Run: `env -u ANTHROPIC_API_KEY -u OPENAI_API_KEY -u GOOGLE_API_KEY -u OPENROUTER_API_KEY PAPERBANANA_LIVE_CLI=1 .venv/bin/python -m unittest tests.test_cli_agents_live -v`
Expected: 3 tests OK. Without the env var: 3 skipped.

- [ ] **Step 3: Document** — `configs/model_config.template.yaml` gets, under `defaults:`, the comment block:

```yaml
  # No API key? Use your logged-in coding-agent CLI instead:
  #   main_model_name: "claude-code/sonnet"   # or "claude-code", "codex", "codex/<model>"
  #   image_gen_model_name: "codex"           # Codex's built-in image generation (Claude Code cannot make images)
```

README: a "Use Claude Code or Codex instead of API keys" subsection with the same model names, the prerequisites (`claude` logged in via `claude /login`, `codex` logged in via `codex login`), the `PAPERBANANA_CLAUDE_BIN`/`PAPERBANANA_CODEX_BIN` overrides, the note that API-key env vars are stripped for these calls so the subscription is used, and that usage counts against the subscription limits.

- [ ] **Step 4: End-to-end run with no API keys**

Create `configs/model_config.yaml` (gitignored) only if absent, with `main_model_name: "claude-code/sonnet"` and `image_gen_model_name: "codex"`. Run the demo agent pipeline on one small diagram input through `skill/run.py` or `main.py` (whichever accepts a single custom input), with all API-key env vars unset. Expected: a JPEG output is produced; view it.

- [ ] **Step 5: Commit**

```bash
git add tests/test_cli_agents_live.py configs/model_config.template.yaml README.md
git commit -m "docs: document Claude Code / Codex CLI bridge; add live smoke test"
```
