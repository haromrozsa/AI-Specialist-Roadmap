"""
MCP client demo — no API key required.
======================================

Normally an MCP server is driven by an LLM client (Claude Desktop, an agent).
To show the full round-trip *without* an API key, this script acts as the client
itself: it launches ``server.py`` as a subprocess over stdio, lists its tools,
then chains them -- ``get_sample`` to fetch a real digit, ``classify_digit`` to
predict it -- and compares the prediction against the known label.

Run from inside this directory::

    python client_demo.py
"""

import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SAMPLE_INDEX = 7  # which digits-dataset sample to fetch and classify


def _payload(result) -> dict:
    """Extract a tool's dict return value from an MCP CallToolResult.

    FastMCP serializes a dict return into a JSON text block; newer SDKs also
    attach it as ``structuredContent``. Parse the text block first, fall back to
    structured content, so this works across SDK versions.
    """
    for block in result.content:
        text = getattr(block, "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        # Some versions wrap scalar/dict returns under a "result" key.
        return structured.get("result", structured)
    raise RuntimeError(f"no parseable payload in tool result: {result!r}")


async def main() -> None:
    # Launch server.py with the same Python interpreter running this client.
    params = StdioServerParameters(command=sys.executable, args=["server.py"])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("[client] tools ->", ", ".join(t.name for t in tools.tools))

            # 1) Fetch a real sample from the server.
            sample = _payload(
                await session.call_tool("get_sample", {"index": SAMPLE_INDEX})
            )
            print(f"[client] get_sample({SAMPLE_INDEX}) -> true_label={sample['true_label']}")

            # 2) Feed its features straight into the classifier tool.
            pred = _payload(
                await session.call_tool(
                    "classify_digit", {"features": sample["features"]}
                )
            )
            print(
                f"[client] classify_digit(...) -> "
                f"prediction={pred['prediction']} confidence={pred['confidence']:.2f}"
            )

            verdict = "MATCH" if pred["prediction"] == sample["true_label"] else "MISMATCH"
            print(
                f"[client] true={sample['true_label']} "
                f"predicted={pred['prediction']} [{verdict}]"
            )


if __name__ == "__main__":
    asyncio.run(main())
