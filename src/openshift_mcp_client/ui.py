"""Gradio UI for the OpenShift MCP chatbot."""

from __future__ import annotations

import atexit
import asyncio

import gradio as gr

from openshift_mcp_client.chatbot import ChatRuntime

_runtime: ChatRuntime | None = None
_lock = asyncio.Lock()

CONSOLE_NAV = """
<aside id="ocp-nav">
  <div id="ocp-brand">
    <svg class="redhat-mark" viewBox="0 0 40 40" width="36" height="36" aria-hidden="true">
      <path fill="#ee0000" d="M20 3c6.2 0 10.6 4.2 11.6 9.4 5.2 1.2 8.4 4.6 8.4 8.8 0 5.2-4.4 8.8-9.6 8.8H9.6C4.4 30 0 26.4 0 21.2c0-4.2 3.2-7.6 8.4-8.8C9.4 7.2 13.8 3 20 3zm-9 24.2h18l-2.2 5.2c-1.8 2.2-4.2 3.4-6.8 3.4s-5-1.2-6.8-3.4L11 27.2z"/>
    </svg>
    <div>
      <div class="openshift">OpenShift</div>
    </div>
  </div>
  <nav aria-label="Assistant">
    <p class="nav-section">Assistant</p>
    <a class="nav-item active" href="#chat-stack">Cluster chat</a>
  </nav>
</aside>
"""

PAGE_HEADING = """
<p id="breadcrumb">OpenShift MCP <span>/</span> Cluster chat</p>
<h1 id="page-title">Cluster chat</h1>
"""

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Red+Hat+Display:wght@400;500;700&family=Red+Hat+Mono:wght@400;500&family=Red+Hat+Text:wght@400;500&display=swap');

html, body, .gradio-container {
  font-family: "Red Hat Text", "Segoe UI", sans-serif !important;
  background: #f0f0f0 !important;
  color: #151515 !important;
}

html, body {
  height: auto;
  min-height: 100%;
  overflow: auto;
}

body {
  display: block !important;
}

.gradio-container,
.gradio-container > .main,
.gradio-container .main > .wrap,
.gradio-container .wrap > .contain {
  max-width: none !important;
  width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
  height: auto !important;
  min-height: 100vh !important;
  max-height: none !important;
  overflow: visible !important;
}

#console {
  display: flex !important;
  flex-wrap: nowrap !important;
  align-items: stretch !important;
  gap: 0 !important;
  width: 100% !important;
  height: auto !important;
  min-height: 100vh !important;
  max-height: none !important;
  margin: 0 !important;
  overflow: visible !important;
}

#nav-slot {
  flex: 0 0 256px !important;
  width: 256px !important;
  max-width: 256px !important;
  min-width: 256px !important;
  min-height: 100vh !important;
  margin: 0 !important;
  padding: 0 !important;
  background: #ffffff !important;
  border: 0 !important;
  border-right: 1px solid #d2d2d2 !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  position: sticky !important;
  top: 0 !important;
  align-self: flex-start !important;
  height: 100vh !important;
  overflow: auto !important;
}

#nav-slot .block,
#nav-slot .html-container,
#nav-slot .prose {
  margin: 0 !important;
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  width: 100% !important;
  max-width: none !important;
  min-width: 0 !important;
}

#ocp-nav {
  width: 256px;
  min-height: 100vh;
  background: #ffffff;
  padding: 18px 0 16px;
}

#ocp-brand {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 0 16px 18px;
  border-bottom: 1px solid #d2d2d2;
}

.openshift {
  font-family: "Red Hat Display", "Red Hat Text", sans-serif;
  font-size: 1.05rem;
  font-weight: 500;
  line-height: 1.15;
  color: #151515;
}

.nav-section {
  margin: 16px 16px 6px;
  font-size: 0.75rem;
  color: #6a6e73;
}

.nav-item {
  display: block;
  margin: 0 8px;
  padding: 8px 12px 8px 14px;
  color: #151515;
  text-decoration: none;
  text-align: left;
  border-radius: 4px;
}

.nav-item.active {
  background: #e7f1fa;
  box-shadow: inset 3px 0 0 #0066cc;
  font-weight: 500;
}

#app-shell {
  flex: 1 1 auto !important;
  max-width: none !important;
  min-width: 0 !important;
  width: auto !important;
  height: auto !important;
  max-height: none !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 16px 24px 24px !important;
  gap: 12px !important;
  box-sizing: border-box !important;
  display: flex !important;
  flex-direction: column !important;
  overflow: visible !important;
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}

#breadcrumb {
  margin: 0;
  font-size: 0.875rem;
  color: #151515;
}

#breadcrumb span {
  color: #6a6e73;
  margin: 0 0.35rem;
}

#page-title {
  margin: 4px 0 0;
  font-family: "Red Hat Display", "Red Hat Text", sans-serif;
  font-size: 1.5rem;
  font-weight: 400;
  line-height: 1.3;
  color: #151515;
}

#page-toolbar {
  align-items: center !important;
  justify-content: space-between !important;
  flex-wrap: nowrap !important;
  gap: 16px !important;
  margin-top: 4px;
  width: 100% !important;
}

#page-toolbar > :first-child {
  flex: 1 1 auto !important;
  min-width: 0 !important;
}

#status-pill {
  font-family: "Red Hat Text", sans-serif !important;
  font-size: 0.875rem;
  border: 0;
  background: transparent;
  color: #6a6e73;
  border-radius: 0;
  padding: 0;
  display: inline-block;
  line-height: 1.4;
}

#status-pill.ready { color: #3e8635; }
#status-pill.error { color: #c9190b; }

#app-shell > :not(#chat-stack) {
  flex: 0 0 auto !important;
}

#chat-stack {
  flex: 0 0 auto !important;
  min-height: 0 !important;
  height: auto !important;
  display: flex !important;
  flex-direction: column !important;
  overflow: visible !important;
  background: #ffffff !important;
  border: 1px solid #d2d2d2 !important;
  border-radius: 1px !important;
  padding: 0 !important;
  gap: 0 !important;
}

#chatbot, .chatbot {
  flex: 0 0 auto !important;
  height: 520px !important;
  min-height: 520px !important;
  max-height: 520px !important;
  overflow: hidden !important;
  border: 0 !important;
  background: #ffffff !important;
  border-radius: 0 !important;
  box-shadow: none !important;
}

#chatbot > .wrapper,
#chatbot .bubble-wrap {
  height: 100% !important;
  min-height: 0 !important;
  max-height: 100% !important;
  flex: 1 1 auto !important;
}

#chatbot .bubble-wrap {
  overflow-x: hidden !important;
  overflow-y: auto !important;
}

#chatbot .message-wrap {
  height: auto !important;
  min-height: 0 !important;
  max-height: none !important;
  flex: 0 0 auto !important;
  justify-content: flex-start !important;
  overflow: visible !important;
  margin-bottom: 4px !important;
}

#chatbot .message,
#chatbot .message-content,
#chatbot .bot,
#chatbot .user,
#chatbot .prose {
  height: auto !important;
  max-height: none !important;
  overflow: visible !important;
  flex: 0 0 auto !important;
}

#chatbot .message-buttons,
#chatbot .icon-button-wrapper {
  margin-top: 0 !important;
}

/* User bubbles use the console selection blue. Assistant bubbles stay white. */
#chatbot .user {
  background-color: #e7f1fa !important;
  color: #151515 !important;
  border-color: #b9dafc !important;
}

#chatbot .bot {
  background-color: #ffffff !important;
  color: #151515 !important;
  border-color: #d2d2d2 !important;
}

#chatbot .user .prose,
#chatbot .user .prose *,
#chatbot .bot .prose,
#chatbot .bot .prose * {
  color: #151515 !important;
}

#chatbot .user .prose a,
#chatbot .bot .prose a {
  color: #0066cc !important;
}

#chatbot .user code,
#chatbot .bot code,
#chatbot .user pre,
#chatbot .bot pre {
  background: #f0f0f0 !important;
  color: #151515 !important;
}

.example {
  background-color: #ffffff !important;
  color: #151515 !important;
  border: 1px solid #d2d2d2 !important;
}

.example .example-text,
.example span {
  color: #151515 !important;
}

.example:hover,
.example:focus {
  background-color: #e7f1fa !important;
  border-color: #0066cc !important;
  color: #151515 !important;
}

.example:hover .example-text,
.example:hover span,
.example:focus .example-text,
.example:focus span {
  color: #151515 !important;
}

#composer {
  flex: 0 0 auto !important;
  align-items: flex-end !important;
  background: #ffffff;
  border-top: 1px solid #d2d2d2;
  padding: 12px 8px 8px;
  margin: 0;
  height: auto !important;
  min-height: 0 !important;
}

#tool-picker,
#message-box {
  border: 0 !important;
  border-radius: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}

#tool-picker label,
#message-box label {
  font-family: "Red Hat Text", sans-serif !important;
  font-size: 0.875rem !important;
  letter-spacing: 0;
  text-transform: none;
  color: #151515 !important;
}

/* Keep the closed multiselect as a single-line control; the option list
   still expands below when the combobox is focused/clicked. */
#tool-picker .wrap {
  background: #ffffff !important;
  border: 1px solid #d2d2d2 !important;
  border-radius: 3px !important;
  min-height: 2.75rem !important;
  max-height: 2.75rem !important;
  overflow: hidden;
}

#tool-picker .wrap-inner {
  flex-wrap: nowrap !important;
  overflow: hidden;
  min-height: 2.75rem;
  max-height: 2.75rem;
  padding-top: 0.45rem !important;
  padding-bottom: 0.45rem !important;
}

#tool-picker .token {
  display: none !important;
}

#tool-picker .secondary-wrap {
  flex: 1 1 auto !important;
  min-width: 0 !important;
}

#tool-picker input[role="combobox"] {
  width: 100% !important;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  cursor: pointer;
}

#tool-picker .remove-all {
  display: none !important;
}

#message-box textarea,
#tool-picker input,
#tool-picker .secondary-wrap {
  background: #ffffff !important;
  color: #151515 !important;
  border-color: #d2d2d2 !important;
}

#message-box textarea {
  border: 1px solid #d2d2d2 !important;
  border-radius: 3px !important;
}

button.primary,
#message-box button.primary,
#message-box button.submit-button {
  background: #0066cc !important;
  border-color: #0066cc !important;
  color: #ffffff !important;
  border-radius: 3px !important;
  font-weight: 500 !important;
}

#message-box button.submit-button:hover {
  background: #004080 !important;
  border-color: #004080 !important;
  color: #ffffff !important;
}

button.secondary {
  background: #ffffff !important;
  color: #151515 !important;
  border: 1px solid #d2d2d2 !important;
  border-radius: 3px !important;
}

.gradio-container .options {
  border: 1px solid #d2d2d2 !important;
  background: #ffffff !important;
  color: #151515 !important;
}

.gradio-container .options .item {
  color: #151515 !important;
}

.gradio-container .options .item.selected,
.gradio-container .options .item:hover,
.gradio-container .options .item.active {
  background: #e7f1fa !important;
}

footer { display: none !important; }

@media (max-width: 760px) {
  #console {
    flex-direction: column !important;
    flex-wrap: nowrap !important;
    height: auto !important;
    max-height: none !important;
    overflow: visible !important;
  }

  #nav-slot {
    flex: 0 0 auto !important;
    align-self: stretch !important;
    width: 100% !important;
    max-width: none !important;
    min-width: 0 !important;
    height: auto !important;
    min-height: 0 !important;
    max-height: none !important;
    position: relative !important;
    border-right: 0 !important;
    border-bottom: 1px solid #d2d2d2 !important;
  }

  #ocp-nav {
    width: 100%;
    min-height: 0;
    padding-bottom: 8px;
  }

  #app-shell {
    flex: 1 1 auto !important;
    width: 100% !important;
    height: auto !important;
    max-height: none !important;
    min-height: 0 !important;
    overflow: visible !important;
  }
}
"""

TOOL_PICKER_JS = """
() => {
  const bind = (root) => {
    if (!root) {
      return false;
    }
    if (root.dataset.summaryBound === "1") {
      return true;
    }
    root.dataset.summaryBound = "1";

    const summarize = () => {
      const input = root.querySelector('input[role="combobox"]');
      if (!input) {
        return;
      }
      const count = root.querySelectorAll(".token").length;
      if (count === 0) {
        input.placeholder = "Select MCP tools…";
        if (document.activeElement !== input) {
          input.value = "";
        }
        return;
      }
      const label = count === 1 ? "1 tool selected" : `${count} tools selected`;
      input.placeholder = label;
      if (document.activeElement !== input) {
        input.value = label;
      }
    };

    summarize();
    new MutationObserver(summarize).observe(root, {
      childList: true,
      subtree: true,
      characterData: true,
    });
    root.addEventListener("focusout", () => window.setTimeout(summarize, 0));
    return true;
  };

  if (bind(document.getElementById("tool-picker"))) {
    return;
  }
  const timer = window.setInterval(() => {
    if (bind(document.getElementById("tool-picker"))) {
      window.clearInterval(timer);
    }
  }, 200);
  window.setTimeout(() => window.clearInterval(timer), 15000);
}
"""

EXAMPLES = [
    {"text": "List pods in the httpd namespace"},
    {"text": "List pods in the default namespace"},
    {"text": "Show the last 100 log lines from pod httpd-ccd6fb678-z95kl in httpd"},
    {"text": "List projects"},
]


async def _get_runtime() -> ChatRuntime:
    global _runtime
    async with _lock:
        if _runtime is None or not _runtime.ready:
            _runtime = await ChatRuntime.create(
                session_id="openshift-gradio",
                db_path="openshift-mcp-client.db",
                clear_history=True,
            )
        return _runtime


async def _shutdown_runtime() -> None:
    global _runtime
    async with _lock:
        if _runtime is not None:
            await _runtime.close()
            _runtime = None


def _shutdown_sync() -> None:
    try:
        asyncio.run(_shutdown_runtime())
    except Exception:
        pass


def _message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        return "".join(parts)
    return ""


def _selected_tool_names(value: object) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item]
    return []


def _stage_user_message(message: str, history: list | None):
    text = (message or "").strip()
    history = list(history or [])
    if not text:
        return gr.skip(), gr.skip()
    history.append({"role": "user", "content": text})
    return "", history


def _stage_example(example: gr.SelectData):
    value = example.value
    text = value.get("text", "") if isinstance(value, dict) else str(value or "")
    text = text.strip()
    if not text:
        return gr.skip()
    return [{"role": "user", "content": text}]


def _prepare_retry(history: list | None) -> list:
    history = list(history or [])
    while history and history[-1].get("role") == "assistant":
        history.pop()
    return history


def _prepare_edit(history: list | None, edit_data: gr.EditData) -> list:
    raw_index = edit_data.index
    index = raw_index[0] if isinstance(raw_index, (list, tuple)) else raw_index
    trimmed = list((history or [])[:index])
    text = edit_data.value if isinstance(edit_data.value, str) else _message_text(
        {"content": edit_data.value}
    )
    trimmed.append({"role": "user", "content": text.strip()})
    return trimmed


def _undo_turn(history: list | None):
    history = list(history or [])
    while history and history[-1].get("role") == "assistant":
        history.pop()
    text = ""
    if history and history[-1].get("role") == "user":
        text = _message_text(history[-1])
        history.pop()
    return history, text


async def _complete_reply(history: list | None, selected_tools: object) -> list:
    history = list(history or [])
    if not history or history[-1].get("role") != "user":
        return history

    text = _message_text(history[-1]).strip()
    names = _selected_tool_names(selected_tools)
    if not text:
        reply = "Ask a question about your OpenShift cluster."
    elif not names:
        reply = "Select at least one MCP tool before asking."
    else:
        try:
            runtime = await _get_runtime()
            reply = await runtime.ask(text, tool_names=names)
        except Exception as exc:  # noqa: BLE001
            reply = f"Something went wrong: {exc}"

    history.append({"role": "assistant", "content": reply})
    return history


async def connect_on_load():
    try:
        runtime = await _get_runtime()
        names = runtime.tool_names
        count = len(names)
        noun = "tool" if count == 1 else "tools"
        status = (
            f'<div id="status-pill" class="ready">● MCP connected · {count} {noun}</div>'
        )
        return status, gr.update(choices=names, value=names, interactive=True)
    except Exception as exc:  # noqa: BLE001
        return (
            f'<div id="status-pill" class="error">○ MCP error: {exc}</div>',
            gr.update(choices=[], value=[], interactive=False),
        )


async def on_clear() -> str:
    try:
        runtime = await _get_runtime()
        await runtime.clear_history()
        return '<div id="status-pill" class="ready">● Session cleared</div>'
    except Exception as exc:  # noqa: BLE001
        return f'<div id="status-pill" class="error">○ Clear failed: {exc}</div>'


def build_theme() -> gr.themes.Base:
    blue = gr.themes.Color(
        c50="#f2f9ff",
        c100="#e7f1fa",
        c200="#b9dafc",
        c300="#73bcf7",
        c400="#2b9af3",
        c500="#0066cc",
        c600="#004d99",
        c700="#004080",
        c800="#002952",
        c900="#001f3f",
        c950="#000d14",
    )
    return gr.themes.Base(
        primary_hue=blue,
        neutral_hue="gray",
        font=[gr.themes.GoogleFont("Red Hat Text"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("Red Hat Mono"), "monospace"],
    ).set(
        body_background_fill="#f0f0f0",
        body_background_fill_dark="#f0f0f0",
        block_background_fill="#ffffff",
        block_background_fill_dark="#ffffff",
        block_border_color="#d2d2d2",
        block_border_color_dark="#d2d2d2",
        block_label_text_color="#151515",
        block_label_text_color_dark="#151515",
        block_title_text_color="#151515",
        block_title_text_color_dark="#151515",
        body_text_color="#151515",
        body_text_color_dark="#151515",
        body_text_color_subdued="#6a6e73",
        body_text_color_subdued_dark="#6a6e73",
        color_accent_soft="#e7f1fa",
        color_accent_soft_dark="#e7f1fa",
        background_fill_primary="#ffffff",
        background_fill_primary_dark="#ffffff",
        background_fill_secondary="#f0f0f0",
        background_fill_secondary_dark="#f0f0f0",
        border_color_accent="#0066cc",
        border_color_accent_dark="#0066cc",
        border_color_primary="#d2d2d2",
        border_color_primary_dark="#d2d2d2",
        input_background_fill="#ffffff",
        input_background_fill_dark="#ffffff",
        input_background_fill_focus="#ffffff",
        input_background_fill_focus_dark="#ffffff",
        button_primary_background_fill="#0066cc",
        button_primary_background_fill_hover="#004080",
        button_primary_text_color="#ffffff",
        button_secondary_background_fill="#ffffff",
        button_secondary_background_fill_hover="#f0f0f0",
        button_secondary_text_color="#151515",
        button_secondary_border_color="#d2d2d2",
        shadow_drop="none",
        shadow_drop_lg="none",
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Cluster chat · OpenShift", fill_width=True) as demo:
        with gr.Row(elem_id="console"):
            with gr.Column(elem_id="nav-slot", scale=0, min_width=256):
                gr.HTML(CONSOLE_NAV)
            with gr.Column(elem_id="app-shell", scale=1, min_width=0):
                gr.HTML(PAGE_HEADING)
                with gr.Row(elem_id="page-toolbar"):
                    status = gr.HTML('<div id="status-pill">○ connecting…</div>')
                    clear_btn = gr.Button(
                        "Clear session",
                        variant="secondary",
                        size="sm",
                        scale=0,
                        min_width=140,
                    )

                with gr.Column(elem_id="chat-stack"):
                    chatbot = gr.Chatbot(
                        elem_id="chatbot",
                        height=520,
                        show_label=False,
                        editable="user",
                        examples=EXAMPLES,
                    )
                    with gr.Row(elem_id="composer"):
                        tools = gr.Dropdown(
                            label="MCP tools",
                            info="Click to toggle tools for the next ask",
                            multiselect=True,
                            filterable=False,
                            allow_custom_value=False,
                            choices=[],
                            value=[],
                            interactive=False,
                            elem_id="tool-picker",
                            scale=2,
                            min_width=240,
                        )
                        message = gr.Textbox(
                            label="Message",
                            placeholder="Ask about pods, namespaces, events, or logs…",
                            autofocus=True,
                            submit_btn="Ask",
                            elem_id="message-box",
                            scale=5,
                            lines=1,
                            max_lines=6,
                        )

                reply_inputs = [chatbot, tools]
                message.submit(
                    _stage_user_message,
                    [message, chatbot],
                    [message, chatbot],
                    queue=False,
                ).then(
                    _complete_reply,
                    reply_inputs,
                    chatbot,
                    show_progress="minimal",
                )
                chatbot.example_select(
                    _stage_example,
                    None,
                    chatbot,
                    queue=False,
                ).then(
                    _complete_reply,
                    reply_inputs,
                    chatbot,
                    show_progress="minimal",
                )
                chatbot.retry(
                    _prepare_retry,
                    chatbot,
                    chatbot,
                    queue=False,
                ).then(
                    _complete_reply,
                    reply_inputs,
                    chatbot,
                    show_progress="minimal",
                )
                chatbot.edit(
                    _prepare_edit,
                    chatbot,
                    chatbot,
                    queue=False,
                ).then(
                    _complete_reply,
                    reply_inputs,
                    chatbot,
                    show_progress="minimal",
                )
                chatbot.undo(_undo_turn, chatbot, [chatbot, message], queue=False)

                demo.load(connect_on_load, outputs=[status, tools])
                clear_btn.click(on_clear, outputs=status)

    return demo


def main() -> None:
    atexit.register(_shutdown_sync)
    demo = build_ui()
    demo.queue()
    demo.launch(
        theme=build_theme(),
        css=CUSTOM_CSS,
        js=TOOL_PICKER_JS,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
    )


if __name__ == "__main__":
    main()
