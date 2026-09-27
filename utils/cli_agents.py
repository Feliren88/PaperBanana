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
