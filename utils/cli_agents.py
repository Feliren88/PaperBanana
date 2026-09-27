"""Run PaperBanana model calls through a logged-in coding-agent CLI.

Model names ``claude-code[/<model>]`` and ``codex[/<model>]`` route a call to
``claude -p`` or ``codex exec`` instead of a provider API, so the user's
Claude Code / Codex subscription login is used and no API key is needed.
Standard library only.
"""

import asyncio
import base64
import json
import os
import shutil
import signal
import tempfile
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
            texts.append(f"[Image: {path.name}]")
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
    if backend == "codex" and image_paths:
        parts.append("The attached images, in order: " + ", ".join(p.name for p in image_paths) + ".")
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


DEFAULT_TIMEOUT = 900  # seconds; Codex image generation takes ~1 minute


def _subprocess_env(backend):
    stripped = BACKENDS[backend][2]
    return {k: v for k, v in os.environ.items() if k not in stripped}


def _kill(proc):
    """Kill the CLI and anything it spawned (it runs in its own process group)."""
    if proc.returncode is not None:
        return
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except ProcessLookupError:
        pass


async def _run(cmd, prompt, *, cwd, env, timeout):
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=cwd, env=env, start_new_session=True,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(prompt.encode()), timeout)
    except asyncio.TimeoutError:
        _kill(proc)
        await proc.wait()
        raise CLIAgentError(f"timed out after {timeout}s")
    except BaseException:
        # Cancelled (e.g. a sibling candidate failed): never leave the CLI running.
        _kill(proc)
        await proc.wait()
        raise
    if proc.returncode != 0:
        # claude -p reports failures such as "Not logged in" as JSON on stdout.
        detail = "\n".join(
            b.decode(errors="replace").strip() for b in (stderr, stdout) if b.strip()
        )
        raise CLIAgentError(f"exited with {proc.returncode}: {detail[-800:]}")
    return stdout.decode(errors="replace")


def _parse_text(backend, stdout, output_file):
    if backend == "claude-code":
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            raise CLIAgentError(f"unexpected output: {stdout[:300]!r}")
        if not isinstance(payload, dict):
            raise CLIAgentError(f"unexpected output: {stdout[:300]!r}")
        if payload.get("is_error"):
            raise CLIAgentError(f"reported an error: {payload.get('result')!r}")
        text = payload.get("result") or ""
        if not isinstance(text, str):
            raise CLIAgentError(f"unexpected result: {text!r}")
    else:
        text = output_file.read_text(encoding="utf-8", errors="replace") if output_file.exists() else ""
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
        except Exception as e:  # any per-call failure keeps the providers' "Error" contract
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
