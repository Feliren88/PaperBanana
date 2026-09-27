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
