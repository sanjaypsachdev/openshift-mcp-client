"""Guardrail model selection without calling a model server."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from openshift_mcp_client.chatbot import (
    ASSISTANT_MODEL,
    BASE_URL,
    build_agent,
    guardrail_target,
)


class GuardrailTargetTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = {
            name: os.environ.get(name)
            for name in ("GUARDRAIL_BASE_URL", "GUARDRAIL_MODEL")
        }
        os.environ.pop("GUARDRAIL_BASE_URL", None)
        os.environ.pop("GUARDRAIL_MODEL", None)

    def tearDown(self) -> None:
        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_unset_values_use_the_assistant_model(self) -> None:
        self.assertEqual(guardrail_target(), (BASE_URL, ASSISTANT_MODEL))

    def test_blank_values_use_the_assistant_model(self) -> None:
        os.environ["GUARDRAIL_BASE_URL"] = "  "
        os.environ["GUARDRAIL_MODEL"] = ""
        self.assertEqual(guardrail_target(), (BASE_URL, ASSISTANT_MODEL))

    def test_model_can_change_without_a_new_endpoint(self) -> None:
        os.environ["GUARDRAIL_MODEL"] = "lite-guardrail"
        self.assertEqual(guardrail_target(), (BASE_URL, "lite-guardrail"))

    def test_both_settings_override_the_assistant(self) -> None:
        os.environ["GUARDRAIL_BASE_URL"] = "https://guardrail.example/v1"
        os.environ["GUARDRAIL_MODEL"] = "lite-guardrail"
        self.assertEqual(
            guardrail_target(),
            ("https://guardrail.example/v1", "lite-guardrail"),
        )


class BuildAgentModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = {
            name: os.environ.get(name)
            for name in ("GUARDRAIL_BASE_URL", "GUARDRAIL_MODEL")
        }
        os.environ.pop("GUARDRAIL_BASE_URL", None)
        os.environ.pop("GUARDRAIL_MODEL", None)

    def tearDown(self) -> None:
        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_same_settings_reuse_the_assistant_model(self) -> None:
        sentinel = "shared-model"
        with patch(
            "openshift_mcp_client.chatbot.build_chat_model",
            return_value=sentinel,
        ) as build:
            agent = build_agent(object())  # type: ignore[arg-type]

        build.assert_called_once_with()
        self.assertIs(agent.model, sentinel)

    def test_separate_settings_build_a_second_model(self) -> None:
        os.environ["GUARDRAIL_BASE_URL"] = "https://guardrail.example/v1"
        os.environ["GUARDRAIL_MODEL"] = "lite-guardrail"
        created: list[str] = []

        def fake_build(
            *,
            base_url: str = BASE_URL,
            model_name: str = ASSISTANT_MODEL,
        ) -> str:
            created.append(f"{model_name}@{base_url}")
            return created[-1]

        with patch(
            "openshift_mcp_client.chatbot.build_chat_model",
            side_effect=fake_build,
        ):
            agent = build_agent(object())  # type: ignore[arg-type]

        self.assertEqual(
            created,
            [
                f"{ASSISTANT_MODEL}@{BASE_URL}",
                "lite-guardrail@https://guardrail.example/v1",
            ],
        )
        self.assertEqual(agent.model, created[0])
        checker = agent.input_guardrails[0].guardrail_function.__closure__
        assert checker is not None
        closed_models = {
            getattr(cell.cell_contents, "model", None) for cell in checker
        }
        self.assertIn(created[1], closed_models)
