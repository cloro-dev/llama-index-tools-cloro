# llama-index-tools-cloro

[LlamaIndex](https://www.llamaindex.ai) tools for
[cloro](https://cloro.dev/?utm_source=llama-index-tools-cloro&utm_medium=byok). cloro loads the live page of a search
engine or an AI assistant for a query and a country, and returns the result as structured data.

| Tool | cloro endpoint | Returns | Credits per call |
|---|---|---|--:|
| `google_search` | `POST /v1/monitor/google` | One Document per Google organic result: title, URL and snippet. With `include_ai_overview=True`, the Google AI Overview and its sources first. | 5, or 7 with the AI Overview |
| `google_ai_mode` | `POST /v1/monitor/aimode` | The Google AI Mode answer as Markdown, with its sources | 6 |
| `bing_ai_mode` | `POST /v1/monitor/bing/aimode` | The Bing AI Mode (Copilot Search in Bing) answer, with its sources | 6 |
| `chatgpt` | `POST /v1/monitor/chatgpt` | The ChatGPT answer that users see, with its sources | 7 |
| `gemini` | `POST /v1/monitor/gemini` | The Gemini answer, with its sources | 6 |

## Installation

```bash
pip install llama-index-tools-cloro
```

Get an API key at [dashboard.cloro.dev](https://dashboard.cloro.dev/api-keys?utm_source=llama-index-tools-cloro&utm_medium=byok).
The free plan includes 500 credits each month. Set the key as `CLORO_API_KEY`, or pass `api_key=...`.

## Usage

### With an agent

```python
import asyncio

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.tools.cloro import CloroToolSpec

cloro = CloroToolSpec(country="US", include_ai_overview=True)
agent = FunctionAgent(
    tools=cloro.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
    system_prompt="Answer with live search results. Cite the URLs you use.",
)


async def main() -> None:
    print(await agent.run("What do Google AI Mode and ChatGPT say are the best SERP APIs?"))


asyncio.run(main())
```

To give the agent only some tools, filter the list:

```python
tools = [t for t in cloro.to_tool_list() if t.metadata.name in ("google_search", "google_ai_mode")]
```

### Direct calls

```python
from llama_index.tools.cloro import CloroToolSpec

cloro = CloroToolSpec()

for doc in cloro.google_search("retrieval-augmented generation", max_results=5, country="GB"):
    print(doc.metadata["title"], doc.metadata["url"])

answer = cloro.google_ai_mode("What is retrieval-augmented generation?")[0]
print(answer.text)
print(answer.metadata["sources"])
```

Each tool returns a list of `Document`. The text is what the agent reads:

- `google_search`: one Document per result. The text is the title, the URL and the snippet. `metadata` has `title`,
  `url` and `position`. Results with an empty link or a Google redirect link are skipped. With the AI Overview
  turned on, the first Document is the AI Overview and its sources, with `metadata["engine"] == "google_ai_overview"`.
- The AI answer tools: one Document. The text is the answer as Markdown, then a "Sources" list. `metadata` has
  `engine`, `prompt` and `sources` (a list of `title` and `url`).

### Parameters

`CloroToolSpec(api_key=None, country="US", timeout=180.0, include_ai_overview=False)`

- `api_key`: the cloro API key. Defaults to the `CLORO_API_KEY` environment variable.
- `country`: ISO 3166-1 alpha-2 country code for the search location. Each tool also takes `country` per call.
- `timeout`: request timeout in seconds. An AI answer can take 15 to 60 seconds.
- `include_ai_overview`: make `google_search` also return the Google AI Overview. It costs 2 more credits per search.

## Limits

- Bing AI Mode accepts a prompt of up to 3,500 characters after URL encoding. Non-Latin scripts allow fewer
  characters. `bing_ai_mode` refuses a longer prompt before it sends the request.
- Your plan sets how many requests can run at the same time. A request sent while all slots are busy gets a `429`
  (`CONCURRENCY_LIMIT_EXCEEDED`).
- An API error raises `CloroError` with the HTTP status, the cloro error code and the message.

## Development

```bash
uv run --with 'llama-index-core>=0.13,<0.15' --with httpx --with pytest -- python -m pytest tests
```

The tests mock the network. The API reference is at
[cloro.dev/docs](https://cloro.dev/docs?utm_source=llama-index-tools-cloro&utm_medium=byok).

## License

MIT
