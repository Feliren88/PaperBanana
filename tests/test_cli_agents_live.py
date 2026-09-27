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
