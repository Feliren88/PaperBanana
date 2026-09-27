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
