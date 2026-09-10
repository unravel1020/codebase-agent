# codebase-agent

An LLM agent that answers questions about a **local code repository** — with tool
calling, RAG retrieval, conversation memory, Pydantic structured output and an
evaluation harness. Built as a small, readable reference for modern AI
application engineering (not an enterprise platform).

* Default provider: **DeepSeek** (`deepseek-chat`), via any **OpenAI-compatible** endpoint.
* No hard-coded secrets: everything comes from `.env` / environment variables.
* Everything except the real LLM call runs **offline** — the test suite, the
  retrieval eval and a full agent-loop demo need **no API key**.

```bash
pip install -r requirements.txt
python -m codebase_agent ask --repo . "How does the path sandbox work?"
```

<details>
<summary>中文速览</summary>

* **做什么**：给一个本地代码仓库路径，用 LLM + 工具调用 + RAG 回答关于代码的问题，并输出结构化结果（summary / relevant_files / evidence / confidence）。
* **默认模型**：DeepSeek（`deepseek-chat`），但只依赖 OpenAI-compatible 接口，`LLM_BASE_URL` + `LLM_MODEL` 可换任何厂商。
* **离线可跑**：pytest 106 通过、retrieval eval、`ask --offline` 全流程都不需要 API key。
* **不需要 key 的命令**：`index` / `search` / `grep` / `read` / `ask --offline`。
* **需要 key 的**：`ask`（真实 LLM）、`evals/run_evals.py --mode llm`。
</details>

---

## Architecture

```
                 ┌──────────────────────────── CLI (src/codebase_agent/cli.py) ────────────────────────────┐
                 │  ask | index | search | grep | read                                                  │
                 └───────────────────────────────────────────┬──────────────────────────────────────────────┘
                                                             │
   ┌───────────────┐   Settings.from_env()   ┌───────────────▼───────────────┐   ChatOpenAI(base_url, model,
   │ .env / env    │────────────────────────▶│  config.Settings (frozen)     │   api_key, timeout, retries)
   │ (no secrets   │                         └───────┬───────────────┬───────┘◀──────────── llm.build_chat_model
   │  in code)     │                                 │               │
   └───────────────┘                                 │               │
                                                     │               │
                        ┌────────────────────────────▼──┐     ┌──────▼────────────────────┐
                        │ Repository (sandbox)          │     │ Embeddings                │
                        │  resolve() -> root check      │     │  local = hashing (offline)│
                        │  read_text() size/binary cap  │     │  openai = any compatible  │
                        │  search() symbol/keyword      │     └──────┬────────────────────┘
                        └───────┬───────────────┬───────┘            │
                                │               │                    │
              ┌─────────────────▼──┐   ┌────────▼─────────┐  ┌───────▼─────────────────┐
              │ tools.py           │   │ rag.CodeRetriever│  │ vectorstore.VectorIndex │
              │  search_code       │   │  chunk + embed   │─▶│  FAISS IndexFlatIP      │
              │  read_file         │   │  top-k retrieve  │  │  (cosine = inner prod)  │
              │  retrieve_context  │◀──┴──────────────────┘  └─────────────────────────┘
              └─────────┬──────────┘
                        │  LangChain BaseTool
              ┌─────────▼──────────────────────────────────────────────┐
              │ agent.CodebaseAgent  (tool-calling loop)              │
              │   llm.bind_tools(tools) -> tool_calls -> ToolMessage  │
              │   memory.ConversationMemory (sliding window)          │
              │   final step: with_structured_output(CodeAnswer)      │
              └─────────┬─────────────────────────────────────────────┘
                        │
              ┌─────────▼─────────────────────┐        ┌───────────────────────────┐
              │ schemas.CodeAnswer (Pydantic) │        │ evals/run_evals.py        │
              │  summary / relevant_files /   │        │  hit rate, MRR, latency,  │
              │  evidence / confidence        │        │  tool-call count -> JSON  │
              └───────────────────────────────┘        └───────────────────────────┘
```

Module responsibilities (one job each):

| Module | Responsibility |
| --- | --- |
| `config.py` | Frozen `Settings` from `.env`/env; provider presets; key masking; `require_api_key()`. |
| `llm.py` | Builds `ChatOpenAI` for any OpenAI-compatible endpoint (base_url, model, timeout, retries). |
| `repository.py` | The sandbox: path resolution, traversal defence, size/binary limits, keyword search, RAG loading. |
| `tools.py` | The three LangChain `@tool`s the model may call; converts repo errors into `ERROR:` strings. |
| `embeddings.py` | `HashingEmbeddings` (offline, deterministic) + OpenAI-compatible embeddings. |
| `vectorstore.py` | FAISS `IndexFlatIP` wrapper with citation metadata. |
| `rag.py` | Chunking (Python-aware), batched embedding, top-k retrieval. |
| `memory.py` | Sliding-window conversation memory. |
| `schemas.py` | `CodeAnswer` / `EvidenceItem` Pydantic schemas + robust JSON extraction. |
| `agent.py` | The tool-calling loop, evidence guards, structured synthesis, audit trail. |
| `offline.py` | `ScriptedChatModel` + `HeuristicPolicy`: run the real loop with no API key. |
| `factory.py` | Wiring: path → repository → retriever → tools → agent. |
| `cli.py` | `ask` / `index` / `search` / `grep` / `read`. |

---

## Agent Loop

`CodebaseAgent.run(question)` (`src/codebase_agent/agent.py`):

1. **Prompt assembly** — `SystemMessage(rules)` + `memory.messages` (sliding window) + `HumanMessage(question)`.
2. **Model step** — `llm.bind_tools(tools).invoke(messages)`. The model returns either
   tool calls or a final message.
3. **Tool step** — every requested tool is executed through the LangChain tool
   interface, timed, recorded in a `ToolCallRecord`, and appended back as a
   `ToolMessage` (keyed by `tool_call_id`).
4. **Loop** — repeat until the model answers, `AGENT_MAX_ITERATIONS` is hit, or
   `AGENT_MAX_TOOL_CALLS` is reached (then the model is told to answer with what it has).
5. **Evidence guard** — if the model answered *without a single tool call* and
   `AGENT_REQUIRE_EVIDENCE=1`, the loop injects a nudge and runs once more. If it
   still has no evidence, confidence is clamped to ≤ 0.2 and a warning is added.
6. **Structured synthesis** — a separate call with
   `llm.with_structured_output(CodeAnswer, method=...)` turns the collected tool
   output into `summary / relevant_files / evidence / confidence`. If the endpoint
   does not support structured output, the response is parsed as JSON text, and
   `confidence` is capped at 0.5 when the answer cites no evidence.
7. **Memory** — the question and answer summary are appended to
   `ConversationMemory` for the next turn.

Properties that make it testable:

* The loop is ~60 lines of explicit control flow — no framework black box.
* `bind_tools` is the only model capability required, so `ScriptedChatModel`
  (a real `BaseChatModel` that replays tool-call decisions) exercises the exact
  production path offline.
* Every step leaves an audit trail (`AgentResult.to_dict()`): tool name, args,
  ok/error, per-call duration, iteration count, total latency.

```
question ──▶ [model] ──tool_calls──▶ [tools] ──ToolMessage──▶ [model] ──▶ answer
                │                                                        │
                └──────────── no tool calls? nudge once ─────────────────┘
                                                                         │
                                        with_structured_output(CodeAnswer) ◀┘
```

---

## RAG Pipeline

```
repository files ──▶ filter (text suffix, not excluded dir, ≤ MAX_FILE_BYTES, not binary)
                 ──▶ chunk (RecursiveCharacterTextSplitter; Python-aware separators for .py)
                 ──▶ metadata: file + start_line/end_line (from add_start_index)
                 ──▶ embed (batches of 64)
                 ──▶ FAISS IndexFlatIP over L2-normalized vectors  (inner product = cosine)
                 ──▶ retrieve(query, k) ──▶ top-k chunks with citations
```

* **Chunking** — `langchain_text_splitters.RecursiveCharacterTextSplitter`.
  `.py`/`.pyi` use `Language.PYTHON` separators (class/def boundaries first);
  everything else uses paragraph/line/word separators. `add_start_index=True` is
  converted into 1-based inclusive line ranges, so retrieved chunks can be cited
  as `file:start-end`.
* **Embeddings** — `EMBEDDING_PROVIDER=local` (default) is a deterministic
  feature-hashing model over word tokens + character 3-grams. It needs no key and
  no network, and works well for identifiers/keywords, which dominate code
  similarity. `EMBEDDING_PROVIDER=openai` switches to any OpenAI-compatible
  `/embeddings` endpoint (`check_embedding_ctx_length=False`, so non-OpenAI
  servers work too). DeepSeek has no public embeddings API, which is why the
  offline default keeps a DeepSeek-only setup fully runnable.
* **Vector store** — FAISS, chosen over Chroma because it is a single wheel with
  no server, no persistence layer and no telemetry: the simplest thing that gives
  real top-k search. `IndexFlatIP` is exact, which is fine for repository-scale
  corpora (this repo: ~165 chunks).
* **Retrieval as a tool** — `retrieve_context` is exposed to the model alongside
  `search_code`/`read_file`, so the agent decides when semantic recall beats
  exact symbol lookup.

---

## Project layout

```
codebase-agent/
├── README.md
├── pyproject.toml            # packaging + pytest config (pythonpath=src)
├── requirements.txt
├── .env.example              # every variable, no secrets
├── .gitignore
├── src/codebase_agent/
│   ├── __init__.py           # public API
│   ├── __main__.py           # python -m codebase_agent
│   ├── config.py
│   ├── llm.py
│   ├── embeddings.py
│   ├── repository.py
│   ├── tools.py
│   ├── vectorstore.py
│   ├── rag.py
│   ├── memory.py
│   ├── schemas.py
│   ├── agent.py
│   ├── offline.py
│   ├── factory.py
│   └── cli.py
├── evals/
│   ├── questions.json        # 10 repository questions + expected files
│   ├── run_evals.py          # retrieval | mock-agent | llm
│   └── results/              # generated JSON reports
└── tests/
    ├── conftest.py           # hermetic env + fixtures
    ├── fixtures/sample_repo/ # tiny app used by tests and offline demos
    ├── test_read_file.py
    ├── test_search_code.py
    ├── test_path_security.py
    ├── test_retrieval.py
    ├── test_structured_output.py
    ├── test_agent_workflow.py
    ├── test_offline.py
    ├── test_memory.py
    ├── test_config.py
    └── test_cli.py
```

---

## Install and run

Requires Python 3.11+ (developed and verified on 3.14.5, Windows).

```bash
cd codebase-agent
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt   # or: pip install -e ".[dev]"
copy .env.example .env            # then set LLM_API_KEY
```

### Configure the API key

The key lives in **`codebase-agent/.env`** (git-ignored, never committed). It is
read by `Settings.from_env()`; nothing is hard-coded and nothing is logged
(`Settings.safe_dict()` masks it as `***xxxx`).

```ini
# .env
LLM_PROVIDER=deepseek
LLM_API_KEY=sk-your-deepseek-key            # <-- the only line you must edit
# DEEPSEEK_API_KEY=sk-...                   # equivalent alternative
# LLM_BASE_URL=https://api.deepseek.com/v1  # optional override
# LLM_MODEL=deepseek-chat                   # optional override
```

Precedence: process environment > `.env` > built-in provider preset. So
`$env:LLM_API_KEY="sk-..."` (PowerShell) or `export LLM_API_KEY=...` also works
and wins over the file.

Then verify (a wrong key prints `401` advice, a wrong URL prints `404` advice —
never a traceback):

```bash
python -m codebase_agent ask --repo tests/fixtures/sample_repo "How are results persisted?"
python -m codebase_agent ask --repo . "How does the agent avoid answering without evidence?" --show-trace
python evals/run_evals.py --mode llm
```

Exit codes: `0` success · `2` configuration problem (missing key) · `3` provider
problem (401 / 404 / connection / rate limit) · `130` interrupted.

Commands (only `ask` without `--offline` needs a key):

```bash
# index a repository and report chunk stats
python -m codebase_agent index --repo .

# symbol/keyword search, no LLM
python -m codebase_agent grep --repo . "CodebaseAgent" -n 10

# read a file with line numbers, no LLM
python -m codebase_agent read --repo . src/codebase_agent/agent.py --start 1 --end 40

# semantic retrieval (RAG) only, no LLM
python -m codebase_agent search --repo . "how is path traversal blocked" -k 5

# full agent loop, heuristic policy, NO API key
python -m codebase_agent ask --repo tests/fixtures/sample_repo "How are results persisted?" --offline --show-trace

# full agent loop with the real model (needs LLM_API_KEY)
python -m codebase_agent ask --repo . "How does the agent avoid answering without evidence?" --json
```

Example (offline, `tests/fixtures/sample_repo`):

```
$ python -m codebase_agent ask --repo tests/fixtures/sample_repo \
    "Where is the average of results computed and how is it persisted?" --offline --show-trace

Summary: The most relevant match ... is app/calculator.py:28 (`def average(values: list[float]) -> float:`).
Confidence: 0.55
Relevant files:
  - app/calculator.py
  ...
-- 2 tool call(s), 3 model iteration(s), 12 ms
   [1] search_code({'query': 'average results computed persisted'}) -> ok 5 ms
   [2] read_file({'path': 'app/calculator.py', 'start_line': 23, 'end_line': 73}) -> ok 1 ms
```

---

## Environment variables

Copy `.env.example` to `.env`. `LLM_BASE_URL` / `LLM_MODEL` always override the
provider preset, so any OpenAI-compatible vendor works.

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_PROVIDER` | `deepseek` | Preset: `deepseek`, `openai`, `openai-compatible`. |
| `LLM_API_KEY` | – | Key for the chat model. Falls back to `DEEPSEEK_API_KEY` / `OPENAI_API_KEY`. Never logged (`safe_dict()` masks it). |
| `LLM_BASE_URL` | provider preset | e.g. `https://api.deepseek.com/v1`. |
| `LLM_MODEL` | provider preset | e.g. `deepseek-chat`. |
| `LLM_TEMPERATURE` | `0` | Keep deterministic for code answers. |
| `LLM_TIMEOUT` | `60` | Per-request timeout (seconds) passed to the client. |
| `LLM_MAX_RETRIES` | `2` | Client-level retries for transient errors. |
| `STRUCTURED_OUTPUT_METHOD` | `function_calling` | `function_calling` (widely supported) or `json_schema`. |
| `EMBEDDING_PROVIDER` | `local` | `local` = offline hashing embeddings, `openai` = compatible `/embeddings`. |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Used when `EMBEDDING_PROVIDER=openai`. |
| `EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY` | – | Override the embedding endpoint/credential. |
| `EMBEDDING_DIM` | `512` | Dimension of the local hashing embeddings. |
| `RAG_TOP_K` | `6` | Retrieved chunks per `retrieve_context` call. |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `1200` / `150` | Chunking parameters. |
| `RAG_MAX_INDEX_FILES` | `400` | Cap on files loaded into the index. |
| `MAX_FILE_BYTES` | `512000` | Files above this are never read or indexed. |
| `MAX_READ_LINES` | `400` | Line budget per `read_file` call. |
| `MAX_SEARCH_RESULTS` | `20` | Line budget per `search_code` call. |
| `AGENT_MAX_TOOL_CALLS` | `8` | Tool-call budget per question. |
| `AGENT_MAX_ITERATIONS` | `6` | Model turns per question. |
| `AGENT_MEMORY_TURNS` | `6` | Sliding-window size of the conversation memory. |
| `AGENT_REQUIRE_EVIDENCE` | `1` | `1` = nudge the model when it answers without tool calls. |
| `REPO_ROOT` | – | Default repository when `--repo` is omitted. |

---

## Testing

```bash
python -m pytest -q            # 106 passed, 1 skipped (Windows symlink test)
```

| Test module | Tests | Covers |
| --- | --- | --- |
| `test_read_file.py` | 12 | whole-file/window reads, line numbering, size + binary + directory rejection, tool error strings |
| `test_search_code.py` | 12 | definition ranking, line numbers, multi-token queries, `max_results`, regex mode, binary/excluded-dir skipping |
| `test_path_security.py` | 13 | `../` traversal (both separators), absolute paths outside root, NUL bytes, empty paths, symlink escape, tool-level blocking |
| `test_retrieval.py` | 13 | chunk metadata + line ranges, determinism, top-k relevance, score ordering, embedding similarity ordering, FAISS validation |
| `test_structured_output.py` | 13 | fenced/raw/prose JSON, invalid payloads, confidence bounds, file normalization, extra-key tolerance, JSON schema |
| `test_agent_workflow.py` | 14 | full tool-calling loop, evidence nudge, confidence clamps, tool budget, unknown tool, failing tool, structured-output fallback, memory across turns |
| `test_offline.py` | 7 | scripted model replay, heuristic policy state machine + reset, offline end-to-end |
| `test_memory.py` | 5 | sliding window, copy semantics, transcript |
| `test_config.py` | 11 | defaults (DeepSeek), presets, overrides, `.env` loading, key fallback, masking, bad values |
| `test_cli.py` | 7 | `index` / `grep` / `read` / `search` / `ask --offline` end-to-end |

No test needs an API key or the network: an autouse fixture clears LLM/embedding
environment variables, and every model interaction goes through
`ScriptedChatModel`.

---

## Evaluation

```bash
python evals/run_evals.py --mode retrieval      # RAG metrics only, no API key
python evals/run_evals.py --mode mock-agent     # full agent loop, offline
python evals/run_evals.py --mode llm            # real model, needs LLM_API_KEY
```

* Questions and expected files live in `evals/questions.json` (10 questions about
  this repository's own source).
* A hit means a returned candidate path equals, or ends with, an expected path.
* Reports are written to `evals/results/<mode>-<UTC timestamp>.json` with the
  full settings (credentials masked), index stats and per-question detail.

Metrics: `hit_rate`, `MRR`, `latency_ms_avg/max`, and for agent modes
`tool_calls_avg/total`, `iterations_avg`, `evidence_rate`, `confidence_avg`.

Recorded on 2026-09-09 against this repository (165 chunks / 41 files, offline
`local` embeddings, `k=6`):

| Mode | Hit rate | MRR | Latency (avg) | Tool calls (avg) | Evidence rate |
| --- | --- | --- | --- | --- | --- |
| `retrieval` | 7/10 = 70% | 0.798 | 0.1 ms | – | – |
| `mock-agent` | 4/10 = 40% | 0.542 | 58.8 ms | 2.00 | 100% |

How to read these numbers:

* `retrieval` is the RAG baseline: 7/10 with purely lexical (hashing) embeddings.
  The 3 misses are paraphrase questions ("FAISS vector index ... normalized")
  where the expected file uses different wording — exactly the gap a real
  embedding model closes.
* `mock-agent` measures the **loop**, not model quality: the heuristic policy
  ranks files by keyword hits only, yet it reaches 100% evidence rate with 2 tool
  calls per question. A real LLM (`--mode llm`) is what raises answer quality.
* The questions file is excluded from the index by default (`--exclude evals`) to
  avoid the questions matching themselves; this alone moved MRR from 0.488 to 0.798.

---

## Design decisions

* **Explicit agent loop over LangGraph.** LangChain 1.x offers
  `langchain.agents.create_agent`, but this project's point is to *show* the loop.
  `bind_tools` + `ToolMessage` + a `while` is the whole agent; swapping in
  `create_agent` is a ~10-line change if you want the framework version.
* **DeepSeek by default, vendor-neutral by construction.** One `ChatOpenAI`
  instance with an explicit `base_url`; `LLM_PROVIDER`/`LLM_BASE_URL`/`LLM_MODEL`
  change vendor without touching code.
* **Local embeddings by default.** DeepSeek has no embeddings API; a deterministic
  offline embedder keeps `pytest`, the eval harness and `ask --offline` fully
  runnable with zero credentials and zero network.
* **Tool errors are strings, not exceptions.** A failing tool returns `ERROR: ...`
  so the loop can recover and stay within budget — and so a malicious model
  cannot crash the process by asking for a bad path.
* **Two-layer path defence.** `Repository.resolve()` is the only door to the
  filesystem: it resolves symlinks, then requires the result to be inside the
  root. Size, binary and directory checks happen before any read.
* **No chain-of-thought.** The structured schema asks for a short factual
  `evidence` note per item plus a confidence score — nothing hidden is requested
  or stored.

---

## Known limitations

1. **Lexical embeddings by default.** The offline hashing embedder has no semantic
   generalization; paraphrase questions miss (see the eval above). Set
   `EMBEDDING_PROVIDER=openai` with a real embedding model for better recall.
2. **`search_code` re-reads files on every call.** Fine for repos up to a few
   hundred files; a large monorepo would want a cached content pass or an
   incremental index. There is also no cross-file symbol index (e.g. "who calls
   this function").
3. **No index persistence.** The FAISS index is rebuilt on every process start
   (fast here: <0.1 s for 41 files). `faiss.write_index` would be the next step.
4. **Memory is a plain sliding window.** No summarization, no per-file memory, no
   retrieval over past turns; long sessions lose early context by design.
5. **One repository per agent, one question at a time.** No multi-agent routing,
   no concurrent tool execution (`parallel_tool_calls` is left to the provider).
6. **Evaluation is small and self-referential.** 10 questions about this repo, one
   expected file each; hit rate is file-level, not answer-level. There is no LLM
   judge and no regression baseline stored in CI.
7. **Structured output depends on the endpoint.** `function_calling` is the
   default because it is the most widely supported mode; providers that support
   neither function calling nor JSON schema fall back to text parsing (with a
   warning in `warnings`).
8. **Symlink escape test is skipped on Windows** unless the process can create
   symlinks (Developer Mode / admin). The check itself is still exercised by the
   `resolve()` tests.
9. **No secrets management beyond env vars.** Fine for local use; a deployed
   version should use a secret manager, and `safe_dict()` only masks, it does not
   encrypt.
10. **Python 3.11+ only.** Verified on 3.14.5; `faiss-cpu`, `langchain-core`,
    `langchain-openai` and `pytest` all resolve to prebuilt wheels there.

## TODO

* [ ] Persist the FAISS index (`save`/`load`) and add an incremental rebuild.
* [ ] Optional `langchain.agents.create_agent` runner behind a flag, to compare
      the explicit loop against the framework agent.
* [ ] Cross-file symbol index for "where is X called" questions.
* [ ] LLM-as-judge scoring in `evals/`, plus a stored baseline to diff against.
* [ ] Streaming output in the CLI, and a `--no-retrieval` mode for pure tool use.
* [ ] GitHub Actions workflow running `pytest` + `run_evals.py --mode retrieval`.

## License

MIT.
