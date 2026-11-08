from unittest.mock import patch

import httpx
import pytest
from llama_index.core.tools.tool_spec.base import BaseToolSpec

from llama_index.tools.cloro import CloroError, CloroToolSpec


def _response(status: int, body: dict) -> httpx.Response:
    return httpx.Response(status, json=body, request=httpx.Request("POST", "https://api.cloro.dev"))


GOOGLE = {
    "success": True,
    "result": {
        "organicResults": [
            {"position": 1, "title": "First", "link": "https://example.com/a", "snippet": "A."},
            {"position": 2, "title": "Unresolved", "link": "", "snippet": "No link."},
            {"position": 3, "title": "Redirect", "link": "https://www.google.com/goto?x=1", "snippet": "R."},
            {"position": 4, "title": "Second", "link": "https://example.org/b", "snippet": "B."},
        ]
    },
}

ANSWER = {
    "success": True,
    "result": {
        "text": "Plain.",
        "markdown": "**Answer.**",
        "sources": [
            {"position": 1, "label": "Source", "url": "https://example.com"},
            {"position": 2, "label": "Again", "url": "https://example.com"},
            {"position": 3, "label": "Search page", "url": "https://www.google.com/search?q=x"},
        ],
    },
}


def test_class():
    assert BaseToolSpec in CloroToolSpec.__mro__
    assert CloroToolSpec.spec_functions == ["google_search", "google_ai_mode", "bing_ai_mode", "chatgpt", "gemini"]


def test_tool_list():
    tools = CloroToolSpec(api_key="k").to_tool_list()
    assert [t.metadata.name for t in tools] == CloroToolSpec.spec_functions


def test_key_from_env(monkeypatch):
    monkeypatch.setenv("CLORO_API_KEY", "env-key")
    assert CloroToolSpec().api_key == "env-key"


def test_missing_key(monkeypatch):
    monkeypatch.delenv("CLORO_API_KEY", raising=False)
    with pytest.raises(ValueError, match="CLORO_API_KEY"):
        CloroToolSpec()


@patch("llama_index.tools.cloro.base.httpx.post")
def test_google_search(post):
    post.return_value = _response(200, GOOGLE)
    docs = CloroToolSpec(api_key="k").google_search("serp api", max_results=5, country="de")

    url = post.call_args.args[0]
    kwargs = post.call_args.kwargs
    assert url == "https://api.cloro.dev/v1/monitor/google"
    assert kwargs["json"] == {"query": "serp api", "country": "DE"}
    assert kwargs["headers"]["Authorization"] == "Bearer k"
    assert [d.metadata["url"] for d in docs] == ["https://example.com/a", "https://example.org/b"]
    assert docs[0].text == "First\nhttps://example.com/a\nA."


@patch("llama_index.tools.cloro.base.httpx.post")
def test_google_search_max_results(post):
    post.return_value = _response(200, GOOGLE)
    assert len(CloroToolSpec(api_key="k").google_search("q", max_results=1)) == 1


@pytest.mark.parametrize(
    ("method", "path", "engine"),
    [
        ("google_ai_mode", "/v1/monitor/aimode", "google_ai_mode"),
        ("bing_ai_mode", "/v1/monitor/bing/aimode", "bing_ai_mode"),
        ("chatgpt", "/v1/monitor/chatgpt", "chatgpt"),
        ("gemini", "/v1/monitor/gemini", "gemini"),
    ],
)
@patch("llama_index.tools.cloro.base.httpx.post")
def test_answers(post, method, path, engine):
    post.return_value = _response(200, ANSWER)
    docs = getattr(CloroToolSpec(api_key="k", country="gb"), method)("What is a SERP?")

    assert post.call_args.args[0] == f"https://api.cloro.dev{path}"
    assert post.call_args.kwargs["json"] == {
        "prompt": "What is a SERP?",
        "country": "GB",
        "include": {"markdown": True},
    }
    assert len(docs) == 1
    assert docs[0].text == "**Answer.**\n\nSources:\n- Source: https://example.com"
    assert docs[0].metadata["engine"] == engine
    assert docs[0].metadata["sources"] == [{"title": "Source", "url": "https://example.com"}]


@patch("llama_index.tools.cloro.base.httpx.post")
def test_bing_prompt_too_long(post):
    # 600 CJK characters are 5,400 characters once URL-encoded.
    with pytest.raises(ValueError, match="too long for Bing AI Mode"):
        CloroToolSpec(api_key="k").bing_ai_mode("中" * 600)
    post.assert_not_called()


@patch("llama_index.tools.cloro.base.httpx.post")
def test_api_error(post):
    post.return_value = _response(
        402, {"success": False, "error": {"code": "INSUFFICIENT_CREDITS", "message": "Not enough credits"}}
    )
    with pytest.raises(CloroError, match="402: INSUFFICIENT_CREDITS Not enough credits"):
        CloroToolSpec(api_key="k").gemini("hi")


@patch("llama_index.tools.cloro.base.httpx.post")
def test_network_error(post):
    post.side_effect = httpx.ConnectError("down")
    with pytest.raises(CloroError, match="failed"):
        CloroToolSpec(api_key="k").chatgpt("hi")
