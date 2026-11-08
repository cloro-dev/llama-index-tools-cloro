"""cloro tool spec: live Google results and AI search engine answers."""

import os
from typing import Any

import httpx
from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

CLORO_API_BASE = "https://api.cloro.dev"
USER_AGENT = "llama-index-tools-cloro"

# Bing AI Mode refuses a prompt longer than this after URL encoding.
BING_PROMPT_LIMIT = 3500


class CloroError(RuntimeError):
    """The cloro API returned an error or a response that cannot be read."""


def _is_direct_link(url: Any) -> bool:
    """Return True for an http(s) link that is not a Google redirect or search page."""
    return (
        isinstance(url, str)
        and url.startswith(("http://", "https://"))
        and "google.com/goto" not in url
        and "google.com/search" not in url
    )


def _encoded_length(prompt: str) -> int:
    from urllib.parse import quote

    return len(quote(prompt.strip(), safe="-_.!~*'()").replace("%20", "+"))


class CloroToolSpec(BaseToolSpec):
    """
    cloro tool spec.

    Live Google search results, and the answers that Google AI Mode, Bing AI Mode,
    ChatGPT and Gemini give to a prompt, with the sources they cite. Each call
    loads the live page for the country you choose and uses cloro credits.
    """

    spec_functions = [
        "google_search",
        "google_ai_mode",
        "bing_ai_mode",
        "chatgpt",
        "gemini",
    ]

    def __init__(
        self,
        api_key: str | None = None,
        country: str = "US",
        timeout: float = 180.0,
    ) -> None:
        """
        Initialize with parameters.

        Args:
            api_key: cloro API key. Defaults to the CLORO_API_KEY environment variable.
            country: ISO 3166-1 alpha-2 country code for the search location.
            timeout: Request timeout in seconds. An AI answer can take 15 to 60 seconds.

        """
        api_key = api_key or os.environ.get("CLORO_API_KEY")
        if not api_key:
            raise ValueError("Set CLORO_API_KEY or pass api_key. Get a key at https://dashboard.cloro.dev")
        self.api_key = api_key
        self.country = country
        self.timeout = timeout

    def google_search(self, query: str, max_results: int = 10, country: str | None = None) -> list[Document]:
        """
        Search Google and return the organic results of the first results page.

        Args:
            query: The search query.
            max_results: The maximum number of results to return.
            country: Optional ISO 3166-1 alpha-2 country code, for example "GB".

        Returns:
            One Document per result. The text has the title, the URL and the snippet.

        """
        result = self._post(
            "/v1/monitor/google",
            {"query": query, "country": (country or self.country).upper()},
        )
        organic = [
            item
            for item in result.get("organicResults") or []
            if isinstance(item, dict) and _is_direct_link(item.get("link"))
        ][:max_results]
        return [
            Document(
                text=f"{item.get('title') or ''}\n{item['link']}\n{item.get('snippet') or ''}".strip(),
                metadata={
                    "title": item.get("title") or "",
                    "url": item["link"],
                    "position": item.get("position"),
                },
            )
            for item in organic
        ]

    def google_ai_mode(self, prompt: str, country: str | None = None) -> list[Document]:
        """
        Ask Google AI Mode a question and return its answer with the cited sources.

        Args:
            prompt: The question to ask.
            country: Optional ISO 3166-1 alpha-2 country code, for example "GB".

        Returns:
            One Document with the answer as Markdown, followed by its sources.

        """
        return self._answer("google_ai_mode", "/v1/monitor/aimode", prompt, country)

    def bing_ai_mode(self, prompt: str, country: str | None = None) -> list[Document]:
        """
        Ask Bing AI Mode (Copilot Search in Bing) a question and return its answer with the cited sources.

        Args:
            prompt: The question to ask. At most 3,500 characters after URL encoding.
            country: Optional ISO 3166-1 alpha-2 country code, for example "GB".

        Returns:
            One Document with the answer as Markdown, followed by its sources.

        """
        if _encoded_length(prompt) > BING_PROMPT_LIMIT:
            raise ValueError(
                "The prompt is too long for Bing AI Mode: at most 3,500 characters after "
                "URL encoding. Non-Latin scripts allow fewer characters."
            )
        return self._answer("bing_ai_mode", "/v1/monitor/bing/aimode", prompt, country)

    def chatgpt(self, prompt: str, country: str | None = None) -> list[Document]:
        """
        Ask ChatGPT a question and return the answer it shows users, with the cited sources.

        Args:
            prompt: The question to ask.
            country: Optional ISO 3166-1 alpha-2 country code, for example "GB".

        Returns:
            One Document with the answer as Markdown, followed by its sources.

        """
        return self._answer("chatgpt", "/v1/monitor/chatgpt", prompt, country)

    def gemini(self, prompt: str, country: str | None = None) -> list[Document]:
        """
        Ask Gemini a question and return its answer with the cited sources.

        Args:
            prompt: The question to ask.
            country: Optional ISO 3166-1 alpha-2 country code, for example "GB".

        Returns:
            One Document with the answer as Markdown, followed by its sources.

        """
        return self._answer("gemini", "/v1/monitor/gemini", prompt, country)

    def _answer(self, engine: str, path: str, prompt: str, country: str | None) -> list[Document]:
        result = self._post(
            path,
            {
                "prompt": prompt,
                "country": (country or self.country).upper(),
                "include": {"markdown": True},
            },
        )
        answer = result.get("markdown") or result.get("text") or ""
        sources: list[dict[str, Any]] = []
        for source in result.get("sources") or []:
            url = source.get("url") if isinstance(source, dict) else None
            if _is_direct_link(url) and url not in [s["url"] for s in sources]:
                sources.append({"title": source.get("label") or "", "url": url})
        text = answer
        if sources:
            text += "\n\nSources:\n" + "\n".join(
                f"- {s['title']}: {s['url']}" if s["title"] else f"- {s['url']}" for s in sources
            )
        return [Document(text=text, metadata={"engine": engine, "prompt": prompt, "sources": sources})]

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = httpx.post(
                f"{CLORO_API_BASE}{path}",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": USER_AGENT,
                },
                timeout=self.timeout,
            )
        except httpx.HTTPError as error:
            raise CloroError(f"Request to cloro {path} failed: {error}") from error
        try:
            data = response.json()
        except ValueError:
            data = None
        if response.is_error or not isinstance(data, dict) or not data.get("success"):
            error = data.get("error") if isinstance(data, dict) else None
            code = error.get("code") if isinstance(error, dict) else None
            message = error.get("message") if isinstance(error, dict) else None
            raise CloroError(
                f"cloro {path} returned HTTP {response.status_code}: {code or 'no error code'} {message or ''}".strip()
            )
        result = data.get("result")
        if not isinstance(result, dict):
            raise CloroError(f"cloro {path} returned no result.")
        return result
