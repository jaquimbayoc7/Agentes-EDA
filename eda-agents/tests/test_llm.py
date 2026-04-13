"""Tests para src/utils/llm.py — utilidad compartida de llamadas a LLM."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.utils.llm import (
    call_claude,
    call_claude_json,
    parse_json_response,
)


# ---------------------------------------------------------------------------
# parse_json_response
# ---------------------------------------------------------------------------


class TestParseJsonResponse:
    def test_json_block(self):
        text = 'Some text\n```json\n{"key": "value"}\n```\nMore text'
        assert parse_json_response(text) == {"key": "value"}

    def test_raw_json(self):
        text = '{"a": 1, "b": 2}'
        assert parse_json_response(text) == {"a": 1, "b": 2}

    def test_embedded_json(self):
        text = 'Here is the result: {"task": "regression"} done.'
        assert parse_json_response(text) == {"task": "regression"}

    def test_fallback_raw(self):
        text = "No JSON here at all."
        assert parse_json_response(text) == {"raw": text}

    def test_json_block_with_list(self):
        text = '```json\n{"items": [1, 2, 3]}\n```'
        result = parse_json_response(text)
        assert result["items"] == [1, 2, 3]


# ---------------------------------------------------------------------------
# call_claude
# ---------------------------------------------------------------------------


class TestCallClaude:
    def test_raises_without_key(self):
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            call_claude("test prompt", api_key="")

    def test_returns_text(self):
        mock_client = MagicMock()
        mock_block = MagicMock()
        mock_block.text = "Hello world"
        mock_client.messages.create.return_value = MagicMock(content=[mock_block])

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        import sys
        original = sys.modules.get("anthropic")
        sys.modules["anthropic"] = mock_anthropic
        try:
            result = call_claude("test", api_key="sk-test-key")
            assert result == "Hello world"
        finally:
            if original is not None:
                sys.modules["anthropic"] = original
            else:
                sys.modules.pop("anthropic", None)

    def test_call_claude_json_parses(self):
        with patch("src.utils.llm.call_claude", return_value='{"key": "val"}'):
            result = call_claude_json("test", api_key="sk-test")
            assert result == {"key": "val"}
