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
            self.assertEqual(text, "first\n\n[Image: input_1.png]\n\n[Image: input_2.jpg]\n\nsecond")
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

    def test_codex_names_attached_images_in_order(self):
        prompt = cli_agents.compose_prompt("codex", "Compare.", [Path("/w/input_1.png"), Path("/w/input_2.jpg")], "")
        self.assertIn("attached images, in order: input_1.png, input_2.jpg", prompt)

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
        "pid": os.getpid(),
    }) + "\n")
calls = len(open(record_path).read().splitlines())
mode = os.environ.get("FAKE_CLI_MODE", "ok")
text = os.environ.get("FAKE_CLI_TEXT", "fake answer")
if mode == "fail_json":
    print(json.dumps({"type": "result", "result": "Not logged in. Please run /login", "is_error": True}))
    sys.exit(1)
if mode == "fail" or (mode == "fail_once" and calls == 1):
    sys.stderr.write("boom")
    sys.exit(2)
if mode == "sleep":
    time.sleep(5)
if args[0] == "exec":
    workdir = Path(args[args.index("-C") + 1])
    Path(args[args.index("-o") + 1]).write_bytes(b"caf\xe9 latin-1" if mode == "latin1" else text.encode("utf-8"))
    image = {"image": PNG_BYTES, "jpeg": JPEG_BYTES, "badimage": b"not an image"}.get(mode)
    if image is not None:
        (workdir / "output.png").write_bytes(image)
elif mode == "json_list":
    print("[]")
elif mode == "nonstring":
    print(json.dumps({"result": 5, "is_error": False}))
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
        self.assertEqual(call["args"][0:2], ["exec", "-i"])
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

    def test_unexpected_json_shapes_become_error(self):
        for mode in ("json_list", "nonstring"):
            with self.subTest(mode=mode):
                self.mode(mode)
                self.assertEqual(self.text("claude-code", max_attempts=1), ["Error"])

    def test_codex_non_utf8_output_is_decoded_not_raised(self):
        self.mode("latin1")
        [answer] = self.text("codex", max_attempts=1)
        self.assertTrue(answer.startswith("caf"))
        self.assertNotEqual(answer, "Error")

    def test_nonzero_exit_reports_cli_json_error_message(self):
        import contextlib, io
        self.mode("fail_json")
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            self.assertEqual(self.text("claude-code", max_attempts=1), ["Error"])
        self.assertIn("Not logged in", log.getvalue())

    def test_cancellation_kills_the_cli_process(self):
        self.mode("sleep")

        async def start_then_cancel():
            task = asyncio.ensure_future(cli_agents.call_cli_agent_async("claude-code", TEXT_CONTENTS, retry_delay=0))
            for _ in range(200):
                if self.calls():
                    break
                await asyncio.sleep(0.02)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(start_then_cancel())
        pid = self.calls()[0]["pid"]
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_claude_image_generation_is_unavailable(self):
        with self.assertRaises(CLIAgentUnavailable):
            self.image("claude-code")
        self.assertEqual(self.calls(), [])

if __name__ == "__main__":
    unittest.main()
