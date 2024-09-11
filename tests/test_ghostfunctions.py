import inspect
from typing import Any
from typing import Dict
from typing import List
from unittest.mock import Mock
from unittest.mock import patch

import openai
import pytest

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
    expected_result = ["returned value from openai"]
    mock_return_result = str(expected_result)

    mock_callable = Mock(
        return_value=openai.openai_object.OpenAIObject.construct_from(
            {"choices": [{"message": {"content": mock_return_result}}]}
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
    expected_result = ["returned value from openai"]
    mock_return_result = str(expected_result)

    mock_callable = Mock(
        return_value=openai.openai_object.OpenAIObject.construct_from(
            {"choices": [{"message": {"content": mock_return_result}}]}
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

    expected_result = ["returned value from openai"]
    mock_return_result = str(expected_result)

    mock_callable = Mock(
        return_value=openai.openai_object.OpenAIObject.construct_from(
            {"choices": [{"message": {"content": mock_return_result}}]}
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

    # More flexible assertions
    call_args = mock_callable.call_args
    assert call_args is not None
    assert call_args.kwargs['messages'] == new_prompt
    assert 'temperature' in call_args.kwargs
    assert isinstance(call_args.kwargs['temperature'], float)

    assert result == expected_result

@pytest.mark.parametrize(
    "expected_result,annotation",
    [
        ("return a string", str),
        (b"return bytes", bytes),
        (1.23, float),
        (11, int),
        (("return", "tuple"), tuple),
        (["return", "list"], List[str]),
        ({"return": "dict"}, Dict[str, str]),
        ({"return", "set"}, set),
        (True, bool),
        (None, None),
    ],
)
def test_ghostfunction_decorator_returns_expected_type(
    expected_result: Any, annotation: Any
) -> None:
    mock_return_result = str(expected_result)

    mock_callable = Mock(
        return_value=openai.openai_object.OpenAIObject.construct_from(
            {"choices": [{"message": {"content": mock_return_result}}]}
        )
    )
    with patch.object(
        ai_ghostfunctions.ghostfunctions,
        "_default_ai_callable",
        return_value=mock_callable,
    ) as patched:

        @ghostfunction
        def generate_n_random_words(n: int, startswith: str) -> annotation:
            """Return a list of `n` random words that start with `startswith`."""
            pass

        result = generate_n_random_words(n=5, startswith="goo")
        patched.assert_called_once()

    assert result == expected_result


def test_ghostfunction_decorator_errors_if_no_return_type_annotation() -> None:
    expected_result = "returned value from openai"

    mock_callable = Mock(
        return_value=openai.openai_object.OpenAIObject.construct_from(
            {"choices": [{"message": {"content": expected_result}}]}
        )
    )

    # test with bare ghostfunction
    with patch.object(
        ai_ghostfunctions.ghostfunctions,
        "_default_ai_callable",
        return_value=mock_callable,
    ):
        with pytest.raises(ValueError):

            @ghostfunction
            def f(a: int):  # type: ignore[no-untyped-def]
                """This is an example that doesn't have a return annotation."""
                pass

    # test with ghostfunction with open-close parens
    with patch.object(
        ai_ghostfunctions.ghostfunctions,
        "_default_ai_callable",
        return_value=mock_callable,
    ):
        with pytest.raises(ValueError):

            @ghostfunction()
            def f2(a: int):  # type: ignore[no-untyped-def]
                """This is an example that doesn't have a return annotation."""
                pass

def test_ghostfunction_default_temperature():
    expected_result = ["returned value from openai"]
    mock_return_result = str(expected_result)

    mock_callable = Mock(
        return_value=openai.openai_object.OpenAIObject.construct_from(
            {"choices": [{"message": {"content": mock_return_result}}]}
        )
    )
    with patch.object(
        ai_ghostfunctions.ghostfunctions,
        "_default_ai_callable",
        return_value=mock_callable,
    ):

        @ghostfunction
        def generate_n_random_words(n: int, startswith: str) -> List[str]:
            """Return a list of `n` random words that start with `startswith`."""
            pass

        result = generate_n_random_words(n=5, startswith="goo")

    call_kwargs = mock_callable.call_args[1]
    assert 'temperature' in call_kwargs
    assert call_kwargs['temperature'] == 0.7  # Default temperature
    assert result == expected_result

def test_truncate_prompt():
    from ai_ghostfunctions.ghostfunctions import truncate_prompt, Message

    long_content = "a" * 10000  # A very long string
    prompt = [
        Message(role="system", content="System message"),
        Message(role="user", content=long_content),
        Message(role="assistant", content="Assistant message"),
    ]

    max_tokens = 100
    truncated_prompt = truncate_prompt(prompt, max_tokens)

    assert len(truncated_prompt) <= 3  # Should not remove the system message
    assert truncated_prompt[0]['role'] == 'system'
    assert len(truncated_prompt[-1]['content']) < len(long_content)  # Content should be truncated


def test_ghostfunction_with_custom_max_tokens():
    expected_result = ["returned value from openai"]
    mock_return_result = str(expected_result)
    custom_max_tokens = 2000

    mock_callable = Mock(
        return_value=openai.openai_object.OpenAIObject.construct_from(
            {"choices": [{"message": {"content": mock_return_result}}]}
        )
    )
    with patch.object(
        ai_ghostfunctions.ghostfunctions,
        "_default_ai_callable",
        return_value=mock_callable,
    ), patch('ai_ghostfunctions.ghostfunctions.truncate_prompt') as mock_truncate:

        @ghostfunction(max_tokens=custom_max_tokens)
        def generate_n_random_words(n: int, startswith: str) -> List[str]:
            """Return a list of `n` random words that start with `startswith`."""
            pass

        result = generate_n_random_words(n=5, startswith="goo")

    mock_truncate.assert_called_once()
    assert mock_truncate.call_args[0][1] == custom_max_tokens
    assert result == expected_result