"""MCP transport selection without connecting to a server."""

from __future__ import annotations

import os
import unittest

from agents.mcp import MCPServer, MCPServerSse, MCPServerStdio, MCPServerStreamableHttp

from openshift_mcp_client.chatbot import build_mcp_server

_ENV_NAMES = ("MCP_TRANSPORT", "MCP_URL", "MCP_COMMAND", "MCP_ARGS")


class McpTransportTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = {name: os.environ.get(name) for name in _ENV_NAMES}

    def tearDown(self) -> None:
        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def _use_env(self, **values: str) -> None:
        for name in _ENV_NAMES:
            os.environ.pop(name, None)
        os.environ.update(values)

    def _assert_common(self, server: MCPServer) -> None:
        self.assertEqual(server.name, "Red Hat OpenShift MCP Server")
        self.assertTrue(server.cache_tools_list)
        self.assertEqual(server.client_session_timeout_seconds, 60)

    def test_unset_transport_uses_native_sse(self) -> None:
        self._use_env(MCP_URL="https://example.test/sse")
        server = build_mcp_server()
        self.assertIsInstance(server, MCPServerSse)
        self.assertEqual(server.params["url"], "https://example.test/sse")
        self._assert_common(server)

    def test_sse_uses_the_native_client(self) -> None:
        self._use_env(MCP_TRANSPORT="sse", MCP_URL="https://example.test/sse")
        server = build_mcp_server()
        self.assertIsInstance(server, MCPServerSse)
        self.assertEqual(server.params["url"], "https://example.test/sse")
        self._assert_common(server)

    def test_streamable_http_uses_the_native_client(self) -> None:
        self._use_env(
            MCP_TRANSPORT="streamable-http",
            MCP_URL="https://example.test/mcp",
        )
        server = build_mcp_server()
        self.assertIsInstance(server, MCPServerStreamableHttp)
        self.assertEqual(server.params["url"], "https://example.test/mcp")
        self._assert_common(server)

    def test_stdio_uses_the_command_and_args(self) -> None:
        self._use_env(
            MCP_TRANSPORT="stdio",
            MCP_COMMAND="npx",
            MCP_ARGS="-y some-server --flag",
        )
        server = build_mcp_server()
        self.assertIsInstance(server, MCPServerStdio)
        self.assertEqual(server.params.command, "npx")
        self.assertEqual(list(server.params.args), ["-y", "some-server", "--flag"])
        self._assert_common(server)

    def test_stdio_without_args(self) -> None:
        self._use_env(MCP_TRANSPORT="stdio", MCP_COMMAND="openshift-mcp")
        server = build_mcp_server()
        self.assertEqual(server.params.command, "openshift-mcp")
        self.assertEqual(list(server.params.args), [])

    def test_url_transports_require_mcp_url(self) -> None:
        cases = (None, "sse", "streamable-http")
        for transport in cases:
            with self.subTest(transport=transport):
                values = {} if transport is None else {"MCP_TRANSPORT": transport}
                self._use_env(**values)
                with self.assertRaises(RuntimeError) as caught:
                    build_mcp_server()
                self.assertIn("MCP_URL", str(caught.exception))

    def test_stdio_requires_mcp_command(self) -> None:
        self._use_env(MCP_TRANSPORT="stdio")
        with self.assertRaises(RuntimeError) as caught:
            build_mcp_server()
        self.assertIn("MCP_COMMAND", str(caught.exception))

    def test_unknown_transport_names_the_allowed_values(self) -> None:
        self._use_env(MCP_TRANSPORT="websocket", MCP_URL="https://example.test")
        with self.assertRaises(RuntimeError) as caught:
            build_mcp_server()
        message = str(caught.exception)
        self.assertIn("sse", message)
        self.assertIn("stdio", message)
        self.assertIn("streamable-http", message)
