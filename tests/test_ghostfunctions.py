import inspect
from typing import List
from unittest.mock import Mock
from unittest.mock import patch

import openai

import ai_ghostfunctions.ghostfunctions
from ai_ghostfunctions import ghostfunction
from ai_ghostfunctions.types import Message


def test_aicallable_function_decorator_has_same_signature() -> None:
    def generate_n_random_words(n: int, startswith: str) -> List[str]:  # type: ignore[empty-body]
        """Return a list of `n` random words that start with `startswith`."""
        pass

    with patch.object(ai_ghostfunctions.ghostfunctions.os, "environ"):  # type: ignore[attr-defined]
        decorated_function = ghostfunction(generate_n_random_words)
        assert inspect.signature(decorated_function) == inspect.signature(
            generate_n_random_words
        )


def test_aicallable_function_decorator() -> None:
    expected_result = "returned value from openai"

    mock_callable = Mock(
        return_value=openai.openai_object.OpenAIObject.construct_from(
            {"choices": [{"message": {"content": expected_result}}]}
        )
    )
    with patch.object(
        ai_ghostfunctions.ghostfunctions,
        "_default_ai_callable",
        return_value=mock_callable,
    ) as patched:

        @ghostfunction
        def generate_n_random_words(n: int, startswith: str) -> List[str]:  # type: ignore[empty-body]
            """Return a list of `n` random words that start with `startswith`."""
            pass

        result = generate_n_random_words(n=5, startswith="goo")
        patched.assert_called_once()

    assert result == expected_result


def test_aicallable_function_decorator_with_open_close_parens() -> None:
    expected_result = "returned value from openai"

    mock_callable = Mock(
        return_value=openai.openai_object.OpenAIObject.construct_from(
            {"choices": [{"message": {"content": expected_result}}]}
        )
    )
    with patch.object(
        ai_ghostfunctions.ghostfunctions,
        "_default_ai_callable",
        return_value=mock_callable,
    ) as patched:

        @ghostfunction()
        def generate_n_random_words(n: int, startswith: str) -> List[str]:  # type: ignore[empty-body]
            """Return a list of `n` random words that start with `startswith`."""
            pass

        result = generate_n_random_words(n=5, startswith="goo")
        patched.assert_called_once()

    assert result == expected_result


def test_aicallable_function_decorator_with_custom_prompt_function() -> None:
    new_prompt = [Message(role="user", content="this is a new prompt")]

    expected_result = "returned value from openai"

    mock_callable = Mock(
        return_value=openai.openai_object.OpenAIObject.construct_from(
            {"choices": [{"message": {"content": expected_result}}]}
        )
    )
    with patch.object(
        ai_ghostfunctions.ghostfunctions,
        "_default_ai_callable",
        return_value=mock_callable,
    ) as patched:

        @ghostfunction(prompt_function=lambda f, **kwargs: new_prompt)
        def generate_n_random_words(n: int, startswith: str) -> List[str]:  # type: ignore[empty-body]
            """Return a list of `n` random words that start with `startswith`."""
            pass

        result = generate_n_random_words(n=5, startswith="goo")
        patched.assert_called_once()
    mock_callable.assert_called_once_with(messages=new_prompt)

    assert result == "returned value from openai"
