"""Printed tool calls are parsed without contacting a model or cluster."""

from __future__ import annotations

import unittest

from openshift_mcp_client.chatbot import (
    coerce_tool_arguments,
    match_tool_name,
    parse_text_tool_call,
    strip_tool_chatter,
)

SELECTED = [
    "pods_get",
    "pods_list",
    "pods_list_in_namespace",
    "pods_log",
]

POD_LOG_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "namespace": {"type": "string"},
        "tail": {"type": "integer"},
    },
    "required": ["name", "namespace"],
}


class ParseTextToolCallTests(unittest.TestCase):
    def test_reads_a_parameters_payload_after_prose(self) -> None:
        text = (
            'I will call the tool to get the logs of the pod '
            '"hello-node-5d7fbb79cc-6t7bh" in the "test" namespace.\n'
            '{"name": "get_pod_logs", "parameters": '
            '{"namespace": "test", "pod": "hello-node-5d7fbb79cc-6t7bh"}}'
        )
        parsed = parse_text_tool_call(text)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        name, arguments = parsed
        self.assertEqual(name, "get_pod_logs")
        self.assertEqual(
            arguments,
            {"namespace": "test", "pod": "hello-node-5d7fbb79cc-6t7bh"},
        )

    def test_reads_a_hermes_arguments_payload(self) -> None:
        text = (
            '{"arguments": {"namespace": "default", "labelSelector": "null", '
            '"fieldSelector": "null"}, "name": "pods_list_in_namespace"}'
        )
        parsed = parse_text_tool_call(text)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed[0], "pods_list_in_namespace")

    def test_leaves_a_normal_answer_alone(self) -> None:
        self.assertIsNone(
            parse_text_tool_call(
                "The pods in the 'test' namespace are:\n"
                "- hello-node-5d7fbb79cc-6t7bh (Running)"
            )
        )


class MatchToolNameTests(unittest.TestCase):
    def test_maps_get_pod_logs_to_pods_log(self) -> None:
        self.assertEqual(match_tool_name("get_pod_logs", SELECTED), "pods_log")

    def test_keeps_an_exact_tool_name(self) -> None:
        self.assertEqual(
            match_tool_name("pods_list_in_namespace", SELECTED),
            "pods_list_in_namespace",
        )


class CoerceToolArgumentsTests(unittest.TestCase):
    def test_renames_pod_and_drops_null_placeholders(self) -> None:
        arguments = coerce_tool_arguments(
            {
                "namespace": "test",
                "pod": "hello-node-5d7fbb79cc-6t7bh",
                "tail": "100",
                "labelSelector": "null",
            },
            POD_LOG_SCHEMA,
        )
        self.assertEqual(
            arguments,
            {
                "namespace": "test",
                "name": "hello-node-5d7fbb79cc-6t7bh",
                "tail": 100,
            },
        )

    def test_turns_previous_false_into_a_boolean(self) -> None:
        arguments = coerce_tool_arguments(
            {"name": "httpd-ccd6fb678-c5nsg", "previous": "false"},
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "previous": {"type": "boolean"},
                    "tail": {"type": "integer", "default": 100},
                },
                "required": ["name"],
            },
        )
        self.assertEqual(
            arguments,
            {
                "name": "httpd-ccd6fb678-c5nsg",
                "previous": False,
                "tail": 100,
            },
        )
        self.assertIs(arguments["previous"], False)
        arguments = coerce_tool_arguments(
            {"name": "httpd-ccd6fb678-c5nsg"},
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "tail": {"type": "integer", "default": 100},
                    "previous": {"type": "boolean"},
                },
                "required": ["name"],
            },
        )
        self.assertEqual(
            arguments,
            {"name": "httpd-ccd6fb678-c5nsg", "tail": 100},
        )

    def test_drops_the_word_pod_as_a_container_name(self) -> None:
        arguments = coerce_tool_arguments(
            {
                "name": "httpd-ccd6fb678-c5nsg",
                "namespace": "test",
                "container": "pod",
            },
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "namespace": {"type": "string"},
                    "container": {"type": "string"},
                },
                "required": ["name"],
            },
        )
        self.assertEqual(
            arguments,
            {"name": "httpd-ccd6fb678-c5nsg", "namespace": "test"},
        )


class StripToolChatterTests(unittest.TestCase):
    def test_drops_parameter_narration_and_keeps_the_logs(self) -> None:
        text = (
            'The previous parameter must be a boolean. However, since the '
            'previous parameter is not provided, I will assume the default '
            'value of "false".\n'
            "\n"
            "Here are the last 100 lines of the 'httpd-ccd6fb678-c5nsg' "
            "pod's logs:\n"
            "\n"
            "[Wed Dec 6 14:30:00 2023] [error] client denied by server "
            "configuration: /var/www/html/\n"
            "[Wed Dec 6 14:30:04 2019] [error] client denied by server "
            "configuration: /var/www/html/\n"
            "\n"
            "The pod's logs show repeated client denied errors.\n"
        )
        cleaned = strip_tool_chatter(text)
        self.assertNotIn("parameter", cleaned.lower())
        self.assertNotIn("100 lines", cleaned.lower())
        self.assertIn("httpd-ccd6fb678-c5nsg", text)
        self.assertIn("[Wed Dec 6 14:30:00 2023]", cleaned)
        self.assertIn("repeated client denied errors", cleaned)
