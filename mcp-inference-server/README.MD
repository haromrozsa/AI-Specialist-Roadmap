# MCP Inference Server (ONNX Digit Classifier)

# Session Summary: A minimal Model Context Protocol (MCP) server that exposes a trained ONNX digit classifier as callable tools, plus a Python MCP client that drives it over stdio — standardizing the tool-calling pattern previously hand-rolled in the ReAct agent demos.

## What I Did During the Session

1. **Built an MCP Server with FastMCP**:
   - Used the official `mcp` Python SDK's high-level `FastMCP` API (`server.py`)
   - Exposed two tools with `@mcp.tool()` decorators, using docstrings + type hints as the tool schema clients see
   - Ran it over **stdio** (`mcp.run(transport="stdio")`), the standard MCP transport where the client launches the server as a subprocess and talks JSON-RPC over stdin/stdout

2. **Exposed a Trained Model as Tools**:
   - `get_sample(index)` — returns a real handwritten-digit sample (`features`, `true_label`) from the sklearn digits dataset, giving an agent something concrete to classify
   - `classify_digit(features)` — normalizes the 64 pixel values by `/16` and runs the ONNX model, returning `prediction`, `confidence`, and full `probabilities`
   - Reused the same `LogisticRegression` model from the `fastapi-onnx` demo (`model.onnx` copied verbatim), loaded once at startup with `onnxruntime.InferenceSession`

3. **Handled the skl2onnx Probability Output Correctly**:
   - `skl2onnx` exports the probability output as a **ZipMap** — `onnxruntime` returns it as a Python list of `{class: prob}` dicts, not a plain array
   - Wrote `_extract_probabilities()` to read that dict and order it by class, computing a real confidence instead of arg-maxing the label output the way `fastapi-onnx/main.py` does
   - Made it robust to both the ZipMap and plain-ndarray export configurations

4. **Wrote a Python MCP Client (no API key)**:
   - `client_demo.py` uses `stdio_client` + `ClientSession` to launch the server, `initialize()` the session, and `list_tools()`
   - Chained the two tools — `get_sample(7)` → feed its `features` into `classify_digit` — and compared the prediction against the known label
   - Added a version-tolerant result parser (`_payload`) that reads the JSON text block first and falls back to `structuredContent`

5. **Documented Real-Client Usage**:
   - Added a `claude_desktop_config.json` snippet so the server can be registered with Claude Desktop and used by an actual LLM ("fetch digit sample 7 and classify it")

---

## What I Learned

1. **What MCP Is and Why It Matters**:
   - The **Model Context Protocol** is an open standard for connecting LLM apps (Claude Desktop, agents, IDEs) to external **tools**, **data**, and **prompts**
   - Instead of hard-wiring a capability into every app, you run one MCP *server* that advertises its tools; any MCP-aware *client* discovers and calls them over a common transport
   - This is the standardized evolution of the manual, regex/Pydantic tool-calling in the [`../langgraph_agent`](../langgraph_agent) ReAct demos

2. **FastMCP Server Basics**:
   - `FastMCP("name")` creates the server; the name is what clients display
   - `@mcp.tool()` turns a plain function into a tool — its **docstring becomes the description** and its **type hints become the input schema**, so good docstrings are functional, not just cosmetic
   - An MCP server is a long-lived process, so load heavy assets (the ONNX session, the dataset) **once at import**, not per call

3. **Writing an MCP Client**:
   - `stdio_client(StdioServerParameters(...))` spawns the server; `ClientSession` wraps the JSON-RPC connection
   - You must `await session.initialize()` before calling anything — this is the MCP handshake
   - Driving a server programmatically (no LLM) is the practical way to **test** an MCP server
   - Tool results arrive as content blocks; a dict return is serialized to a JSON text block (and, on newer SDKs, `structuredContent`) — parse defensively across versions

4. **stdio Transport Mechanics**:
   - The client owns the server's lifecycle: it launches `server.py` as a subprocess and communicates over stdin/stdout
   - Spawning with `sys.executable` guarantees the server runs under the same interpreter as the client
   - Loading `model.onnx` via `Path(__file__).parent` keeps it working regardless of the client's working directory

5. **Exposing an Existing Model as a Capability**:
   - Wrapping an already-trained ONNX model as MCP tools required no retraining — just a thin, well-described tool surface and correct preprocessing (`/16` normalization matching training)
   - Splitting into `get_sample` + `classify_digit` lets an agent chain tools, which is the real payoff of the protocol over a single monolithic endpoint

---

## How to Run

**Dependencies** (`mcp` needs **Python ≥ 3.10**): `pip install mcp onnxruntime scikit-learn numpy`

**Demo (no API key):**

```bash
cd mcp-inference-server
python client_demo.py
```

Expected output:

```
[client] tools -> get_sample, classify_digit
[client] get_sample(7) -> true_label=3
[client] classify_digit(...) -> prediction=3 confidence=0.98
[client] true=3 predicted=3 [MATCH]
```

**Use from Claude Desktop** — add to `claude_desktop_config.json` (absolute paths), then restart:

```json
{
  "mcpServers": {
    "digit-inference": {
      "command": "python",
      "args": ["/absolute/path/to/mcp-inference-server/server.py"]
    }
  }
}
```

---

This session provided hands-on experience turning a trained model into an agent-usable capability behind the Model Context Protocol — building both the server (FastMCP tools over stdio) and a client that drives it — and reinforced how MCP standardizes the tool-calling pattern that earlier demos implemented by hand.
