"""Shared OpenShift MCP chatbot runtime used by the CLI and Gradio UI."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from agents import (
    Agent,
    InputGuardrailTripwireTriggered,
    MaxTurnsExceeded,
    ModelSettings,
    OpenAIChatCompletionsModel,
    OutputGuardrailTripwireTriggered,
    RunConfig,
    Runner,
    SQLiteSession,
    set_tracing_disabled,
)
from agents.extensions.tool_output_trimmer import ToolOutputTrimmer
from agents.items import ToolCallItem, ToolCallOutputItem
from agents.mcp import (
    MCPServer,
    MCPServerSse,
    MCPServerStdio,
    MCPServerStreamableHttp,
    create_static_tool_filter,
)
from agents.result import RunResult
from dotenv import load_dotenv
from openai import AsyncOpenAI, BadRequestError

from openshift_mcp_client.guardrails import (
    OFF_TOPIC_INPUT_REPLY,
    OFF_TOPIC_OUTPUT_REPLY,
    drop_blocked_user_message,
    make_openshift_guardrails,
    store_output_refusal,
)

set_tracing_disabled(True)

# .env is the project configuration and replaces stale values inherited from the shell.
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set. Add it to the .env file in the project root."
        )
    return value


def _optional_env(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


BASE_URL = _required_env("BASE_URL")
ASSISTANT_MODEL = _required_env("ASSISTANT_MODEL")

AGENT_INSTRUCTIONS = (
    "You are an OpenShift assistant.\n"
    "1. Call exactly one of the tools available to you to answer the user.\n"
    "2. After the tool returns, reply immediately with the facts.\n"
    "3. Quote pod names, statuses, namespaces, and log lines from tool output.\n"
    "4. Never say you will run a command again.\n"
    "5. Never invent cluster data.\n"
    "6. If none of the available tools can answer, say which tool is needed.\n"
    "7. Stay on OpenShift. If the question is about something else, do not call "
    "a tool. Say you can only help with OpenShift.\n"
    "8. Use the tool interface. Never write a JSON function call in your reply.\n"
    "9. Pass boolean arguments as true or false and integer arguments as numbers. "
    "When a property has a default, pass that default unless the user asked for "
    "another value.\n"
    "10. Never mention tools, parameters, defaults, types, or omitted arguments. "
    "The reply is only the cluster answer.\n"
    "11. Omit container unless the user names a specific container. The word pod "
    "is not a container name."
)

HEDGE_PHRASES = (
    "run the command again",
    "i will run",
    "let me run",
    "does not show any additional",
    "i'll check again",
)


def _tool_output_text(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
            elif getattr(item, "text", None):
                parts.append(str(item.text))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(output, dict):
        if "text" in output:
            return str(output["text"])
        return json.dumps(output, indent=2)
    return str(output)


_TOOL_NAME_STOPWORDS = {
    "a",
    "an",
    "describe",
    "fetch",
    "for",
    "get",
    "in",
    "list",
    "of",
    "read",
    "show",
    "the",
}
_ARGUMENT_ALIASES = {
    "pod": "name",
    "podname": "name",
    "pod_name": "name",
    "ns": "namespace",
    "taillines": "tail",
    "tail_lines": "tail",
    "tailLines": "tail",
}


def _balanced_json(text: str, start: int) -> str | None:
    opening = text[start]
    closing = "}" if opening == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _json_values(text: str) -> list[Any]:
    values: list[Any] = []
    repaired = text.replace('"name: "', '"name": "')
    for index, char in enumerate(repaired):
        if char not in "{[":
            continue
        blob = _balanced_json(repaired, index)
        if blob is None:
            continue
        try:
            values.append(json.loads(blob))
        except json.JSONDecodeError:
            continue
    return values


def _argument_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if isinstance(value, dict):
        return value
    return None


def parse_text_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    """Return a function call printed as text, not sent through the tool API."""
    if not text or '"name"' not in text.replace('"name: "', '"name": "'):
        return None
    for value in _json_values(text):
        items = value if isinstance(value, list) else [value]
        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            arguments = _argument_dict(item.get("arguments", item.get("parameters")))
            if isinstance(name, str) and name.strip() and arguments is not None:
                return name.strip(), arguments
    return None


def _tool_name_stems(name: str) -> set[str]:
    stems: set[str] = set()
    for part in re.split(r"[^a-z0-9]+", name.lower()):
        if not part or part in _TOOL_NAME_STOPWORDS:
            continue
        if part.endswith("s") and not part.endswith("ss") and len(part) > 3:
            part = part[:-1]
        stems.add(part)
    return stems


def match_tool_name(requested: str, available: list[str]) -> str | None:
    """Map a printed tool name onto one selected MCP tool."""
    if requested in available:
        return requested
    by_lower = {name.lower(): name for name in available}
    if requested.lower() in by_lower:
        return by_lower[requested.lower()]
    wanted = _tool_name_stems(requested)
    if not wanted:
        return None
    ranked: list[tuple[float, float, str]] = []
    for name in available:
        have = _tool_name_stems(name)
        if not have:
            continue
        overlap = wanted & have
        if not overlap:
            continue
        cover = len(overlap) / len(wanted)
        score = len(overlap) / len(wanted | have)
        if cover < 1 and score < 0.67:
            continue
        ranked.append((cover, score, name))
    if not ranked:
        return None
    ranked.sort(reverse=True)
    if len(ranked) > 1 and ranked[0][:2] == ranked[1][:2]:
        return None
    return ranked[0][2]


def _nullish(value: Any) -> bool:
    return value is None or (
        isinstance(value, str) and value.strip().lower() in {"", "null", "none"}
    )


def _coerce_schema_value(value: Any, schema: dict[str, Any]) -> Any:
    kind = schema.get("type")
    if kind == "integer" and isinstance(value, str):
        stripped = value.strip()
        if stripped.lstrip("-").isdigit():
            return int(stripped)
    if kind == "boolean" and isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
    return value


_NOT_CONTAINER_NAMES = {
    "container",
    "containers",
    "log",
    "logs",
    "pod",
    "pods",
}


def _drop_bogus_container(target: str, value: Any, cleaned: dict[str, Any]) -> bool:
    if target != "container" or not isinstance(value, str):
        return False
    text = value.strip()
    if text.lower() in _NOT_CONTAINER_NAMES:
        return True
    pod_name = cleaned.get("name")
    return isinstance(pod_name, str) and text == pod_name.strip()


def coerce_tool_arguments(
    arguments: dict[str, Any],
    schema: dict[str, Any] | None,
) -> dict[str, Any]:
    """Drop null placeholders and rename aliases to the tool's real fields."""
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    if not isinstance(properties, dict):
        properties = {}
    cleaned: dict[str, Any] = {}
    for key, value in arguments.items():
        if _nullish(value):
            continue
        target = key
        if properties and key not in properties:
            alias = _ARGUMENT_ALIASES.get(key) or _ARGUMENT_ALIASES.get(key.lower())
            if (
                alias
                and alias in properties
                and alias not in cleaned
                and _nullish(arguments.get(alias))
            ):
                target = alias
            else:
                continue
        field_schema = properties.get(target, {})
        if not isinstance(field_schema, dict):
            field_schema = {}
        value = _coerce_schema_value(value, field_schema)
        if field_schema.get("type") == "boolean" and not isinstance(value, bool):
            continue
        if _drop_bogus_container(target, value, cleaned):
            continue
        cleaned[target] = value
    for key, field_schema in properties.items():
        if key in cleaned or not isinstance(field_schema, dict):
            continue
        if "default" not in field_schema or _nullish(field_schema.get("default")):
            continue
        cleaned[key] = _coerce_schema_value(field_schema["default"], field_schema)
    return cleaned


_TOOL_CHATTER = re.compile(
    r"\b("
    r"parameters?|arguments?|default value|not provided|"
    r"must be a boolean|i will assume|i'll assume|input schema"
    r")\b",
    re.IGNORECASE,
)
_REQUESTED_LINE_COUNT = re.compile(r"\blast\s+\d+\s+lines\b", re.IGNORECASE)


def _looks_like_log_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("[") or (
        stripped.startswith("I") and "log.go" in stripped
    )


def _is_tool_chatter(text: str) -> bool:
    return bool(_TOOL_CHATTER.search(text) or _REQUESTED_LINE_COUNT.search(text))


def strip_tool_chatter(text: str) -> str:
    """Remove sentences that describe the tool instead of the cluster."""
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if _looks_like_log_line(stripped):
            kept.append(line.rstrip())
            continue
        parts = re.split(r"(?<=[.!?])\s+", stripped)
        useful = [part for part in parts if part and not _is_tool_chatter(part)]
        if useful:
            kept.append(" ".join(useful))
    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _call_name_and_arguments(item: ToolCallItem) -> tuple[str, dict[str, Any]]:
    raw = item.raw_item
    if isinstance(raw, dict):
        name = str(raw.get("name") or item.tool_name or "tool")
        arguments = raw.get("arguments", raw.get("input", {}))
    else:
        name = str(getattr(raw, "name", None) or item.tool_name or "tool")
        arguments = getattr(raw, "arguments", None)
        if arguments is None:
            arguments = getattr(raw, "input", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    return name, arguments


def latest_tool_turn(result: RunResult) -> tuple[str, dict[str, Any], str] | None:
    """Return the last completed tool call and its text output."""
    name = "tool"
    arguments: dict[str, Any] = {}
    output = ""
    for item in result.new_items:
        if isinstance(item, ToolCallItem):
            name, arguments = _call_name_and_arguments(item)
        elif isinstance(item, ToolCallOutputItem):
            text = _tool_output_text(item.output).strip()
            if text:
                output = text
    if not output:
        return None
    return name, arguments, output


def format_assistant_reply(result: RunResult) -> str:
    """Prefer a clear answer; if the small model hedges, show tool output."""
    text = strip_tool_chatter((result.final_output or "").strip())
    tool_outputs = [
        _tool_output_text(item.output).strip()
        for item in result.new_items
        if isinstance(item, ToolCallOutputItem)
    ]
    tool_outputs = [t for t in tool_outputs if t]

    hedgy = bool(text) and any(p in text.lower() for p in HEDGE_PHRASES)
    printed_call = parse_text_tool_call(result.final_output or "") is not None
    if tool_outputs and (not text or hedgy or printed_call):
        latest = tool_outputs[-1]
        if hedgy:
            return (
                "Here is the cluster data from the MCP tool:\n\n"
                f"{latest}"
            )
        return latest

    return text or "(No response from the model.)"


def _install_argument_coercion(
    server: MCPServer,
    schemas: dict[str, dict[str, Any]],
) -> None:
    """Convert argument types before the MCP server sees them.

    The log tool rejects ``previous: "false"``. A missing boolean uses the
    server default, so a value that is still not a boolean is left out.
    """
    original = server.call_tool

    async def call_tool(
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
    ):
        prepared = coerce_tool_arguments(arguments or {}, schemas.get(tool_name))
        if meta is None:
            result = await original(tool_name, prepared)
        else:
            result = await original(tool_name, prepared, meta=meta)
        text = _mcp_tool_text(result)
        retry = False
        match = re.search(
            r"(\w+) parameter must be a boolean",
            text,
            re.IGNORECASE,
        )
        if match and match.group(1) in prepared:
            prepared.pop(match.group(1))
            retry = True
        elif "container" in prepared and re.search(
            r"is not valid for pod|not a valid container",
            text,
            re.IGNORECASE,
        ):
            prepared.pop("container")
            retry = True
        if retry:
            if meta is None:
                return await original(tool_name, prepared)
            return await original(tool_name, prepared, meta=meta)
        return result

    server.call_tool = call_tool  # type: ignore[method-assign]


def _mcp_tool_text(result: Any) -> str:
    content = getattr(result, "content", None)
    if content:
        return _tool_output_text(content).strip()
    return _tool_output_text(result).strip()


def _tool_input_schema(tool: Any) -> dict[str, Any]:
    schema = getattr(tool, "inputSchema", None)
    if schema is None:
        schema = getattr(tool, "input_schema", None)
    return schema if isinstance(schema, dict) else {}


async def summarize_tool_result(
    question: str,
    tool_name: str,
    output: str,
    arguments: dict[str, Any] | None = None,
) -> str:
    """Ask the model to answer from a tool result without printing another call."""
    body = output.strip() or "(The tool returned no data.)"
    if len(body) > 6000:
        body = body[:6000] + "\n…"
    called_with = json.dumps(arguments or {}, ensure_ascii=False)
    client = AsyncOpenAI(base_url=BASE_URL, api_key="dummy")
    try:
        response = await client.chat.completions.create(
            model=ASSISTANT_MODEL,
            temperature=0,
            max_tokens=800,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an OpenShift assistant. Answer the user from the "
                        "tool result only. Use the call context to quote the pod, "
                        "namespace, or container that was queried. Do not mention "
                        "tools, parameters, defaults, types, or omitted arguments. "
                        "Do not say how many lines were requested. Describe only "
                        "what is in the result. Do not invent names. Do not print "
                        "a function call or JSON."
                    ),
                },
                {"role": "user", "content": question},
                {
                    "role": "user",
                    "content": (
                        "Call context, not for the user to read:\n"
                        f"{tool_name} {called_with}\n\n"
                        f"Result:\n{body}"
                    ),
                },
            ],
        )
    except Exception:  # noqa: BLE001 - show the tool result if the summary call fails
        return body
    message = response.choices[0].message.content if response.choices else None
    summary = strip_tool_chatter((message or "").strip())
    if not summary or parse_text_tool_call(summary) is not None:
        return body
    return summary


def _latest_tool_output_from_exception(exc: MaxTurnsExceeded) -> str | None:
    for attr in ("run_data", "result", "state"):
        result = getattr(exc, attr, None)
        if result is None:
            continue
        items = getattr(result, "new_items", None) or []
        for item in reversed(items):
            if isinstance(item, ToolCallOutputItem):
                text = _tool_output_text(item.output).strip()
                if text:
                    return text
    return None


def resolve_selected_tools(
    available: list[str],
    selected: list[str] | None,
) -> list[str]:
    """None keeps every discovered tool. A list keeps that intersection."""
    if selected is None:
        return list(available)
    allowed = {name.strip() for name in selected if name and name.strip()}
    return [name for name in available if name in allowed]


_MCP_SERVER_NAME = "Red Hat OpenShift MCP Server"
_MCP_TRANSPORTS = ("sse", "stdio", "streamable-http")


def build_mcp_server() -> MCPServer:
    """Connect with the native transport named by MCP_TRANSPORT.

    An unset MCP_TRANSPORT uses SSE.
    """
    transport = (_optional_env("MCP_TRANSPORT") or "sse").lower()
    if transport == "sse":
        return MCPServerSse(
            name=_MCP_SERVER_NAME,
            params={"url": _required_env("MCP_URL")},
            cache_tools_list=True,
            client_session_timeout_seconds=60,
        )
    if transport == "streamable-http":
        return MCPServerStreamableHttp(
            name=_MCP_SERVER_NAME,
            params={"url": _required_env("MCP_URL")},
            cache_tools_list=True,
            client_session_timeout_seconds=60,
        )
    if transport == "stdio":
        args = shlex.split(_optional_env("MCP_ARGS") or "")
        return MCPServerStdio(
            name=_MCP_SERVER_NAME,
            params={"command": _required_env("MCP_COMMAND"), "args": args},
            cache_tools_list=True,
            client_session_timeout_seconds=60,
        )
    allowed = ", ".join(_MCP_TRANSPORTS)
    raise RuntimeError(f"MCP_TRANSPORT must be {allowed}.")


def guardrail_target() -> tuple[str, str]:
    """Model endpoint for topic checks. Unset values use the assistant model."""
    return (
        _optional_env("GUARDRAIL_BASE_URL") or BASE_URL,
        _optional_env("GUARDRAIL_MODEL") or ASSISTANT_MODEL,
    )


def build_chat_model(
    *,
    base_url: str = BASE_URL,
    model_name: str = ASSISTANT_MODEL,
) -> OpenAIChatCompletionsModel:
    client = AsyncOpenAI(base_url=base_url, api_key="dummy")
    return OpenAIChatCompletionsModel(
        model=model_name,
        openai_client=client,
    )


def build_agent(mcp_server: MCPServer) -> Agent:
    model = build_chat_model()
    guardrail_base_url, guardrail_model_name = guardrail_target()
    if guardrail_base_url == BASE_URL and guardrail_model_name == ASSISTANT_MODEL:
        guardrail_model = model
    else:
        guardrail_model = build_chat_model(
            base_url=guardrail_base_url,
            model_name=guardrail_model_name,
        )
    topic_input, topic_output = make_openshift_guardrails(guardrail_model)
    return Agent(
        name="OpenShift Assistant",
        instructions=AGENT_INSTRUCTIONS,
        model=model,
        model_settings=ModelSettings(
            temperature=0,
            parallel_tool_calls=False,
            tool_choice="auto",
        ),
        mcp_servers=[mcp_server],
        tools=[],
        input_guardrails=[topic_input],
        output_guardrails=[topic_output],
    )


def build_run_config() -> RunConfig:
    return RunConfig(
        call_model_input_filter=ToolOutputTrimmer(
            recent_turns=1,
            max_output_chars=2000,
            preview_chars=400,
        ),
    )


class ChatRuntime:
    """Long-lived MCP + agent session for CLI or Gradio."""

    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self.mcp_server: MCPServer | None = None
        self.agent: Agent | None = None
        self.session: SQLiteSession | None = None
        self.run_config = build_run_config()
        self.tool_names: list[str] = []
        self.tool_schemas: dict[str, dict[str, Any]] = {}
        self._active_tools: tuple[str, ...] | None = None
        self._call_lock = asyncio.Lock()
        self.ready = False

    @classmethod
    async def create(
        cls,
        *,
        session_id: str = "openshift-chat",
        db_path: str = "openshift-mcp-client.db",
        clear_history: bool = True,
    ) -> ChatRuntime:
        runtime = cls()
        await runtime.start(
            session_id=session_id,
            db_path=db_path,
            clear_history=clear_history,
        )
        return runtime

    async def start(
        self,
        *,
        session_id: str = "openshift-chat",
        db_path: str = "openshift-mcp-client.db",
        clear_history: bool = True,
    ) -> None:
        self.mcp_server = build_mcp_server()
        await self._stack.enter_async_context(self.mcp_server)
        tools = await self.mcp_server.list_tools()
        self.tool_names = [tool.name for tool in tools]
        self.tool_schemas = {
            tool.name: _tool_input_schema(tool) for tool in tools
        }
        _install_argument_coercion(self.mcp_server, self.tool_schemas)
        self._apply_tool_selection(self.tool_names)
        self.session = SQLiteSession(
            session_id=session_id,
            db_path=db_path,
            session_settings={"limit": 12},
        )
        if clear_history:
            await self.session.clear_session()
        self.ready = True

    async def close(self) -> None:
        self.ready = False
        await self._stack.aclose()

    async def clear_history(self) -> None:
        if self.session is not None:
            await self.session.clear_session()

    def _apply_tool_selection(self, selected: list[str]) -> None:
        """Expose only the chosen MCP tools."""
        key = tuple(selected)
        if key == self._active_tools and self.agent is not None:
            return
        if self.mcp_server is None:
            return

        self.mcp_server.tool_filter = create_static_tool_filter(
            allowed_tool_names=list(selected),
        )
        self.agent = build_agent(self.mcp_server)
        self._active_tools = key

    async def ask(self, message: str, tool_names: list[str] | None = None) -> str:
        if not self.ready or self.agent is None or self.session is None:
            return "Chat runtime is not connected to the OpenShift MCP server."

        text = message.strip()
        if not text:
            return "Ask a question about your OpenShift cluster."

        selected = resolve_selected_tools(self.tool_names, tool_names)
        if not selected:
            return "Select at least one MCP tool before asking."

        try:
            async with self._call_lock:
                self._apply_tool_selection(selected)
                agent = self.agent
                session = self.session
                if agent is None or session is None:
                    return "Chat runtime is not connected to the OpenShift MCP server."
                try:
                    result = await Runner.run(
                        agent,
                        text,
                        session=session,
                        run_config=self.run_config,
                        max_turns=5,
                    )
                except InputGuardrailTripwireTriggered:
                    try:
                        await drop_blocked_user_message(session, text)
                    except Exception:  # noqa: BLE001 - still tell the user the question was blocked
                        pass
                    return OFF_TOPIC_INPUT_REPLY
                except OutputGuardrailTripwireTriggered:
                    try:
                        await store_output_refusal(session, OFF_TOPIC_OUTPUT_REPLY)
                    except Exception:  # noqa: BLE001 - still tell the user the reply was blocked
                        pass
                    return OFF_TOPIC_OUTPUT_REPLY
        except BadRequestError as exc:
            return (
                "The model rejected that request (likely context length). "
                f"Details: {exc}"
            )
        except MaxTurnsExceeded as exc:
            fallback = _latest_tool_output_from_exception(exc)
            if fallback:
                return (
                    "I hit the turn limit while using tools. "
                    f"Latest tool output:\n\n{fallback}"
                )
            return (
                "I hit the turn limit before finishing. "
                "Please try a simpler question."
            )
        except Exception as exc:  # noqa: BLE001 - surface errors in the UI
            return f"Something went wrong: {exc}"

        final_text = (result.final_output or "").strip()
        printed = parse_text_tool_call(final_text)
        if printed is not None:
            return await self._answer_printed_tool_call(text, printed, selected)

        turn = latest_tool_turn(result)
        if turn is not None:
            tool_name, arguments, output = turn
            arguments = coerce_tool_arguments(
                arguments,
                self.tool_schemas.get(tool_name),
            )
            summary = await summarize_tool_result(
                text,
                tool_name,
                output,
                arguments,
            )
            try:
                await store_output_refusal(session, summary)
            except Exception:  # noqa: BLE001 - still show the summary if history repair fails
                pass
            return summary

        return format_assistant_reply(result)

    async def _answer_printed_tool_call(
        self,
        question: str,
        printed: tuple[str, dict[str, Any]],
        selected: list[str],
    ) -> str:
        """Run a tool the model wrote as JSON, then answer from its result."""
        requested, arguments = printed
        matched = match_tool_name(requested, selected)
        if matched is None or self.mcp_server is None or self.session is None:
            names = ", ".join(selected)
            return (
                f"I could not run a tool named {requested}. "
                f"Selected tools: {names}."
            )
        try:
            result = await self.mcp_server.call_tool(
                matched,
                coerce_tool_arguments(arguments, self.tool_schemas.get(matched)),
            )
        except Exception as exc:  # noqa: BLE001 - tell the user the tool failed
            return f"The {matched} tool failed: {exc}"
        output = _mcp_tool_text(result)
        called_with = coerce_tool_arguments(
            arguments,
            self.tool_schemas.get(matched),
        )
        summary = await summarize_tool_result(
            question,
            matched,
            output,
            called_with,
        )
        try:
            await store_output_refusal(self.session, summary)
        except Exception:  # noqa: BLE001 - still show the summary if history repair fails
            pass
        return summary
