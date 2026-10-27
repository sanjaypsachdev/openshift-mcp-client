"""Topic guardrail behavior that does not need the cluster or the model."""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import AsyncMock, patch

from agents import Agent, RunContextWrapper
from agents.exceptions import ModelBehaviorError
from openai import APIStatusError

from openshift_mcp_client.guardrails import (
    OpenShiftTopicCheck,
    classify_topic,
    drop_blocked_user_message,
    enable_structured_topic_check,
    input_excerpt,
    make_openshift_guardrails,
    output_excerpt,
    store_output_refusal,
    topic_check_from_text,
)


class _MemorySession:
    def __init__(self, items: list[Any]) -> None:
        self.items = list(items)

    async def pop_item(self) -> Any:
        if not self.items:
            return None
        return self.items.pop()

    async def add_items(self, items: list[Any]) -> None:
        self.items.extend(items)


class _Result:
    def __init__(self, output: Any) -> None:
        self.final_output = output


def _status_error(message: str, status_code: int) -> APIStatusError:
    import httpx2

    request = httpx2.Request("POST", "https://example.invalid/v1/chat/completions")
    response = httpx2.Response(status_code, request=request)
    return APIStatusError(message, response=response, body=None)


class ExcerptTests(unittest.TestCase):
    def test_input_excerpt_keeps_the_latest_user_message_and_previous_turn(self) -> None:
        excerpt = input_excerpt(
            [
                {"role": "user", "content": "List pods in the httpd namespace"},
                {
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "httpd-abc is Running"}],
                },
                {"type": "function_call", "name": "pods_log"},
                {"role": "user", "content": "show its logs"},
            ]
        )

        self.assertIn("Earlier user: List pods in the httpd namespace", excerpt)
        self.assertIn("Earlier assistant: httpd-abc is Running", excerpt)
        self.assertIn("Latest user message:\nshow its logs", excerpt)
        self.assertNotIn("pods_log", excerpt)

    def test_output_excerpt_keeps_the_end_of_a_long_reply(self) -> None:
        excerpt = output_excerpt(("a" * 2000) + "TAIL")
        self.assertTrue(excerpt.endswith("TAIL"))
        self.assertIn("\n...\n", excerpt)


class TopicParseTests(unittest.TestCase):
    def test_reads_json_and_one_word_decisions(self) -> None:
        parsed = topic_check_from_text(
            '```json\n{"is_openshift": false, "reasoning": "poem"}\n```'
        )
        self.assertFalse(parsed.is_openshift)
        self.assertEqual(parsed.reasoning, "poem")
        self.assertTrue(topic_check_from_text("ALLOW").is_openshift)
        self.assertFalse(topic_check_from_text("BLOCK").is_openshift)
        self.assertEqual(topic_check_from_text("hello there").reasoning, "unparsed")


class SessionRepairTests(unittest.IsolatedAsyncioTestCase):
    async def test_drops_only_the_matching_user_message(self) -> None:
        session = _MemorySession(
            [
                {"role": "assistant", "content": "pod list"},
                {"role": "user", "content": "write a poem"},
            ]
        )
        await drop_blocked_user_message(session, "write a poem")
        self.assertEqual(session.items, [{"role": "assistant", "content": "pod list"}])

    async def test_restores_the_tail_when_it_is_not_the_blocked_message(self) -> None:
        session = _MemorySession([{"role": "assistant", "content": "pod list"}])
        await drop_blocked_user_message(session, "write a poem")
        self.assertEqual(session.items, [{"role": "assistant", "content": "pod list"}])

    async def test_replaces_a_blocked_assistant_reply(self) -> None:
        session = _MemorySession(
            [
                {"role": "user", "content": "list pods"},
                {"role": "assistant", "content": "The capital of France is Paris."},
            ]
        )
        await store_output_refusal(session, "OpenShift only.")
        self.assertEqual(
            session.items,
            [
                {"role": "user", "content": "list pods"},
                {"role": "assistant", "content": "OpenShift only."},
            ],
        )


class GuardrailRunTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        enable_structured_topic_check(True)

    async def test_input_guardrail_trips_on_an_off_topic_message(self) -> None:
        run = AsyncMock(
            return_value=_Result(
                OpenShiftTopicCheck(is_openshift=False, reasoning="poem")
            )
        )
        with patch("openshift_mcp_client.guardrails.Runner.run", run):
            guardrail, _output = make_openshift_guardrails("test-model")
            result = await guardrail.run(
                Agent(name="OpenShift Assistant"),
                "Write a poem about the ocean",
                RunContextWrapper(context=None),
            )

        self.assertTrue(result.output.tripwire_triggered)
        self.assertFalse(result.output.output_info.is_openshift)
        self.assertEqual(guardrail.run_in_parallel, False)

    async def test_output_guardrail_allows_cluster_data(self) -> None:
        run = AsyncMock(
            return_value=_Result(
                OpenShiftTopicCheck(is_openshift=True, reasoning="pod list")
            )
        )
        with patch("openshift_mcp_client.guardrails.Runner.run", run):
            _input, guardrail = make_openshift_guardrails("test-model")
            result = await guardrail.run(
                RunContextWrapper(context=None),
                Agent(name="OpenShift Assistant"),
                "httpd-abc 1/1 Running",
            )

        self.assertFalse(result.output.tripwire_triggered)

    async def test_classifier_falls_back_when_structured_output_is_rejected(self) -> None:
        structured = Agent(name="structured")
        plain = Agent(name="plain")

        async def fake_run(agent: Agent[Any], *_args: Any, **_kwargs: Any) -> _Result:
            if agent is structured:
                raise _status_error("response_format json_schema is not supported", 400)
            return _Result("BLOCK")

        with patch(
            "openshift_mcp_client.guardrails.Runner.run",
            side_effect=fake_run,
        ):
            check = await classify_topic(structured, plain, "Write a poem")

        self.assertFalse(check.is_openshift)
        with patch(
            "openshift_mcp_client.guardrails.Runner.run",
            AsyncMock(return_value=_Result("ALLOW")),
        ):
            second = await classify_topic(structured, plain, "list pods")
        self.assertTrue(second.is_openshift)

    async def test_classifier_reads_a_decision_buried_in_a_model_error(self) -> None:
        structured = Agent(name="structured")
        plain = Agent(name="plain")
        run = AsyncMock(
            side_effect=ModelBehaviorError(
                'Invalid JSON: {"is_openshift": true, "reasoning": "logs"}'
            )
        )
        with patch("openshift_mcp_client.guardrails.Runner.run", run):
            check = await classify_topic(structured, plain, "show logs")

        self.assertTrue(check.is_openshift)
        self.assertEqual(run.await_count, 1)


if __name__ == "__main__":
    unittest.main()
