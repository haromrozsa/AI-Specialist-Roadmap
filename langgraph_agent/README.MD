# LangGraph: ReAct Agent with Tools

## Session Summary: Built a ReAct (Reasoning + Acting) agent using LangGraph and LangChain that can decide which tools to use, execute them, and loop until a final answer is reached.

---

## What I Did During the Session

1. **Built a ReAct Agent with Manual Parsing**:
    - Created `AgentState` with fields for query, reasoning, tool selection, tool results, and iteration tracking
    - Implemented tool functions (`search_tool`, `calculator_tool`) and a tools registry
    - Built reasoning node that prompts LLM and parses output using regex to extract tool name and input
    - Implemented conditional looping: `reasoning` → `tool_executor` → back to `reasoning` until done

2. **Refactored to Use Pydantic for Structured Output**:
    - Defined `ToolDecision` Pydantic model with typed fields (`tool`, `tool_input`, `reasoning`, `final_answer`)
    - Used `PydanticOutputParser` to automatically generate format instructions for the LLM
    - Replaced manual regex parsing with `parser.parse(response)` for type-safe validation

3. **Implemented LangChain's Built-in Agent**:
    - Defined tools using `@tool` decorator — framework reads docstrings as descriptions
    - Used `create_react_agent()` to create agent with automatic parsing and routing
    - Wrapped in `AgentExecutor` which handles the reasoning loop automatically
    - Reduced code significantly — no manual parsing, routing, or loop management needed

---

## What I Learned

1. **ReAct Pattern (Reasoning + Acting)**:
    - Agent thinks → decides tool → executes → observes result → thinks again
    - Implemented as a loop in LangGraph using conditional edges back to reasoning node
    - Max iterations prevent infinite loops

2. **Three Approaches to Agent Implementation**:

   | Approach | Parsing | Routing | Loop | Code Complexity |
   |----------|---------|---------|------|-----------------|
   | Manual (regex) | You write regex | You write routing function | You build graph edges | High |
   | Pydantic | `PydanticOutputParser` | You write routing function | You build graph edges | Medium |
   | LangChain Built-in | Framework handles | Framework handles | `AgentExecutor` handles | Low |

3. **Tool Definition Patterns**:
    - **Manual**: Dict with function + description string
    - **Pydantic**: Same, but LLM outputs structured JSON
    - **LangChain**: `@tool` decorator — docstring becomes description automatically

4. **LLM Output → Action Flow**:
    - LLM only outputs text (or JSON) — it cannot call functions directly
    - Your code (or framework) parses the output to extract tool name and input
    - Your code (or framework) calls the actual Python function
    - Framework approaches hide this complexity but do the same thing underneath

5. **Pydantic for LLM Output**:
    - `PydanticOutputParser.get_format_instructions()` generates JSON schema instructions
    - `parser.parse(response)` validates and converts JSON string to typed Python object
    - Provides type safety and automatic validation vs manual string parsing

---

This session demonstrated the progression from manual agent implementation to framework-assisted approaches, showing how LangChain abstracts away parsing, routing, and looping while achieving the same ReAct behavior.