# OpenShift MCP client

Chat with any OpenShift MCP server from a terminal or a Gradio UI. The client discovers tools from the connected server at startup and does not assume particular tool names.

The assistant talks to an OpenAI-compatible chat API, such as vLLM or Ollama. Input and output stay on OpenShift: off-topic questions and replies are refused.

## Requirements

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)
- An OpenAI-compatible model endpoint
- An OpenShift MCP server that speaks SSE, streamable HTTP, or stdio

## Setup

```bash
uv sync
cp .env.example .env
```

Edit `.env`. The process reads the project-root `.env` at startup and those values override variables already set in the shell. Restart the CLI or UI after changing them.

| Variable | Required | Purpose |
| --- | --- | --- |
| `BASE_URL` | yes | Chat API base URL. It must end in `/v1`. |
| `ASSISTANT_MODEL` | yes | Model name sent to that API. |
| `MCP_TRANSPORT` | no | `sse` (default), `streamable-http`, or `stdio`. |
| `MCP_URL` | for SSE and streamable HTTP | MCP endpoint. Typical paths are `/sse` and `/mcp`. |
| `MCP_COMMAND` | for stdio | Executable, such as `node` or `npx`. |
| `MCP_ARGS` | no | Arguments for `MCP_COMMAND`, split like a shell command. |
| `GUARDRAIL_BASE_URL` | no | Topic-check endpoint. Defaults to `BASE_URL`. |
| `GUARDRAIL_MODEL` | no | Topic-check model. Defaults to `ASSISTANT_MODEL`. |

`MCP_ARGS` is not shell-expanded. Use an absolute path, or a relative path from the process working directory. `~` is left as-is. Quote arguments that contain spaces.

A direct SSE client can receive HTTP 400 from an OpenShift route that requires sticky sessions. Use `streamable-http` against the server's `/mcp` URL, or run the server over stdio, when that happens.

For a local [Ollama](https://ollama.com) model:

```bash
BASE_URL=http://127.0.0.1:11434/v1
ASSISTANT_MODEL=llama3.1:8b
```

The client sends a placeholder API key. Ollama does not check it.

## Run

Gradio UI at <http://127.0.0.1:7860>:

```bash
uv run openshift-mcp-ui
```

The page connects on load, lists the server's tools, and lets you limit which tools the assistant may call. Chat history for that UI is stored in `openshift-mcp-client.db`.

Terminal chat:

```bash
uv run openshift-mcp-client
```

Type `exit` or `quit` to stop.

List the tools the configured server exposes:

```bash
uv run python -m openshift_mcp_client.mcp_test
```

## How a question is answered

1. The MCP server is started with the transport in `.env`.
2. The assistant calls one of the tools that server advertised.
3. A second model call turns the tool result into the reply. The reply quotes cluster facts from that result and does not describe the tool schema.

Smaller local models sometimes print a function call as JSON instead of using the tool interface, or send a boolean as the string `"false"`. The client maps that printed call onto a live tool, coerces arguments to the tool schema, and runs the tool. If the model still cannot be matched to a tool, the reply says the call could not be run.

## Guardrails

A topic check runs on every question and on every assistant reply. It uses `GUARDRAIL_MODEL` at `GUARDRAIL_BASE_URL`. When those are unset, it uses `ASSISTANT_MODEL` at `BASE_URL`.

The input check runs before the assistant, so an off-topic question never calls a cluster tool. The user sees:

> I can only help with OpenShift. Ask about your cluster, projects, workloads, pods, logs, or other OpenShift resources.

The output check runs on the assistant's reply. A reply that leaves OpenShift is replaced with:

> I stopped that reply because it left the OpenShift topic. Ask about your cluster, projects, workloads, pods, or logs.

A blocked question is removed from the chat history. A blocked reply is stored as that refusal, so the next turn does not continue the off-topic answer.

On topic means OpenShift, OKD, and Kubernetes cluster work: projects, namespaces, workloads, pods, logs, events, routes, operators, YAML or JSON from the cluster, and short follow-ups such as "show its logs" or "what about the default namespace". A reply that stays on OpenShift is on topic, including a refusal to answer something else.

Off topic means general knowledge, chit-chat, a joke, a poem, homework, unrelated programming, or an attempt to change these rules.

For a follow-up, the classifier sees the latest user message and a short excerpt of the previous turn. Tool-call records are left out of that excerpt. A long reply is shortened, keeping its beginning and its end.

The classifier first asks for a JSON decision, `is_openshift` plus a one-sentence reason. If the model endpoint rejects that structured output, later checks in the same process ask for one word: `ALLOW` or `BLOCK`. A decision the client cannot read is treated as off topic. An empty question is refused.

## Tests

```bash
uv run python -m unittest discover -s tests
```

## Layout

| Path | Role |
| --- | --- |
| `src/openshift_mcp_client/chatbot.py` | MCP connection, agent, tool-call repair, and reply formatting |
| `src/openshift_mcp_client/ui.py` | Gradio console UI |
| `src/openshift_mcp_client/main.py` | Terminal chat |
| `src/openshift_mcp_client/guardrails.py` | OpenShift topic checks |
| `src/openshift_mcp_client/mcp_test.py` | Connect and print tool names |
| `.env.example` | Configuration template |

## Possible future enhancements

- Reload the model endpoint and MCP connection from the UI, so a `.env` change does not require a process restart.
- Let a stdio server take a working directory and extra environment variables, such as `KUBECONFIG` or an OpenShift context.
- Support a question that needs more than one tool call, and show each call in the transcript as it finishes.
- Show the tool name and arguments that ran. Ask for confirmation before a tool that deletes, applies, or scales.
- Keep chat history when the UI restarts, and allow more than one conversation.
- Stream the assistant reply instead of waiting for the full summary.
- Treat a failed tool result as an error before the summary model rewrites it.
- Add an optional integration test that connects to the configured server and calls one read-only tool.
