"""
LangGraph Tutorial: ReAct Agent with Pydantic Parsing
======================================================
This example demonstrates:
1. Pydantic models for structured LLM output
2. Automatic parsing with LangChain's PydanticOutputParser
3. Type-safe tool selection
"""

from typing import TypedDict, Literal, Annotated, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import operator


# =============================================================================
# 1. PYDANTIC MODELS FOR STRUCTURED OUTPUT
# =============================================================================

class ToolDecision(BaseModel):
    """
    Structured output from the LLM's reasoning.
    The LLM must respond in this exact format.
    """
    tool: Literal["search", "calculator", "none"] = Field(
        description="The tool to use: 'search', 'calculator', or 'none' if no tool needed"
    )
    tool_input: str = Field(
        description="The input to pass to the tool (empty string if tool is 'none')"
    )
    reasoning: str = Field(
        description="Brief explanation of why this tool was chosen"
    )
    final_answer: Optional[str] = Field(
        default=None,
        description="If tool is 'none', provide the final answer here"
    )


# =============================================================================
# 2. TOOLS DEFINITION (same as before)
# =============================================================================

def search_tool(query: str) -> str:
    """Simulated search tool."""
    knowledge = {
        "python": "Python is a high-level programming language created by Guido van Rossum in 1991.",
        "langchain": "LangChain is a framework for building applications with large language models.",
        "langgraph": "LangGraph is a library for building stateful, multi-agent applications with LLMs.",
        "eiffel tower": "The Eiffel Tower is 330 meters tall and located in Paris, France.",
    }

    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return value
    return "No specific information found for this query."


def calculator_tool(expression: str) -> str:
    """Simple calculator tool."""
    try:
        import re
        clean_expr = re.sub(r'[^0-9+\-*/().\s]', '', expression)
        result = eval(clean_expr)
        return f"The result of {clean_expr} is {result}"
    except Exception as e:
        return f"Could not calculate: {expression}. Error: {str(e)}"


TOOLS = {
    "search": search_tool,
    "calculator": calculator_tool,
}


# =============================================================================
# 3. STATE DEFINITION
# =============================================================================

class AgentState(TypedDict):
    query: str
    tool_decision: Optional[ToolDecision]  # Now using Pydantic model!
    tool_result: str
    iteration: int
    final_answer: str
    history: Annotated[list, operator.add]


# =============================================================================
# 4. LLM + PARSER SETUP
# =============================================================================

# Create the Pydantic parser
parser = PydanticOutputParser(pydantic_object=ToolDecision)


class MockStructuredLLM:
    """
    Mock LLM that returns JSON matching our Pydantic schema.
    """

    def invoke(self, prompt: str) -> str:
        prompt_lower = prompt.lower()

        # Return valid JSON based on query type
        if any(op in prompt for op in ['+', '-', '*', '/']) or "calculate" in prompt_lower:
            import re
            numbers = re.findall(r'[\d+\-*/\s()]+', prompt)
            expr = ''.join(numbers).strip() or "0"
            return f'''{{
    "tool": "calculator",
    "tool_input": "{expr}",
    "reasoning": "This query requires mathematical calculation",
    "final_answer": null
}}'''

        if any(word in prompt_lower for word in ["what is", "who is", "tell me", "search"]):
            # Extract the query topic
            query = prompt.split("Query:")[-1].split("\n")[0].strip() if "Query:" in prompt else "unknown"
            return f'''{{
    "tool": "search",
    "tool_input": "{query}",
    "reasoning": "This query requires searching for information",
    "final_answer": null
}}'''

        if "tool_result" in prompt_lower:
            return '''{{
    "tool": "none",
    "tool_input": "",
    "reasoning": "I have enough information to answer",
    "final_answer": "Based on my research, here is the answer."
}}'''

        return '''{{
    "tool": "none",
    "tool_input": "",
    "reasoning": "I can answer this directly",
    "final_answer": "I don't have specific information about this."
}}'''


def create_llm():
    print("🤖 Using MockStructuredLLM (returns JSON)")
    return MockStructuredLLM()


# =============================================================================
# 5. NODE DEFINITIONS
# =============================================================================

def create_reasoning_node(llm):
    """
    Reasoning node with Pydantic parsing.
    """

    # The parser generates format instructions automatically!
    format_instructions = parser.get_format_instructions()

    prompt_template = PromptTemplate(
        template="""You are a helpful assistant with access to these tools:

1. search - Search for factual information about a topic
2. calculator - Calculate math expressions

Query: {query}

Previous tool result: {tool_result}

Decide which tool to use (or 'none' if you can answer directly).

{format_instructions}
""",
        input_variables=["query", "tool_result"],
        partial_variables={"format_instructions": format_instructions}
    )

    def reasoning_node(state: AgentState) -> dict:
        print(f"\n🧠 REASONING NODE (iteration {state['iteration'] + 1})")

        # Build prompt
        prompt = prompt_template.format(
            query=state["query"],
            tool_result=state.get("tool_result", "None yet")
        )

        # Get LLM response (JSON string)
        response = llm.invoke(prompt)
        print(f"   LLM Response (JSON): {response[:80]}...")

        # Parse with Pydantic - automatic validation!
        try:
            tool_decision = parser.parse(response)
            print(f"   ✅ Parsed successfully: tool={tool_decision.tool}")
        except Exception as e:
            print(f"   ❌ Parse error: {e}")
            # Fallback
            tool_decision = ToolDecision(
                tool="none",
                tool_input="",
                reasoning="Parse error, ending",
                final_answer="Sorry, I encountered an error."
            )

        return {
            "tool_decision": tool_decision,
            "iteration": state["iteration"] + 1,
            "history": [f"Iteration {state['iteration'] + 1}: {tool_decision.tool}"]
        }

    return reasoning_node


def tool_executor_node(state: AgentState) -> dict:
    """
    Executes the selected tool.
    """
    print(f"\n🔧 TOOL EXECUTOR NODE")

    decision = state["tool_decision"]
    tool_name = decision.tool
    tool_input = decision.tool_input

    print(f"   Executing: {tool_name}({tool_input})")
    result = TOOLS[tool_name](tool_input)
    print(f"   Result: {result}")

    return {
        "tool_result": result,
        "history": [f"Tool result: {result[:50]}..."]
    }


def final_answer_node(state: AgentState) -> dict:
    """
    Formats the final answer.
    """
    print(f"\n✅ FINAL ANSWER NODE")

    decision = state["tool_decision"]

    if decision and decision.final_answer:
        answer = decision.final_answer
    elif state["tool_result"]:
        answer = state["tool_result"]
    else:
        answer = "Unable to find an answer."

    formatted = f"""
{'='*60}
🤖 Agent Response (Pydantic Parsing)
{'='*60}
Query: {state['query']}

Answer: {answer}

Reasoning: {decision.reasoning if decision else 'N/A'}
Steps taken: {state['iteration']}
{'='*60}
"""

    return {"final_answer": formatted}


# =============================================================================
# 6. ROUTING FUNCTIONS
# =============================================================================

def should_use_tool(state: AgentState) -> Literal["tool_executor", "final_answer"]:
    """Route based on Pydantic model - type safe!"""

    if state["iteration"] >= 3:
        return "final_answer"

    decision = state["tool_decision"]

    # Type-safe access to the tool field
    if decision.tool in ["search", "calculator"]:
        return "tool_executor"
    else:
        return "final_answer"


def after_tool(state: AgentState) -> Literal["reasoning", "final_answer"]:
    if state["iteration"] >= 3:
        return "final_answer"
    return "reasoning"


# =============================================================================
# 7. GRAPH CONSTRUCTION
# =============================================================================

def build_agent_graph(llm):
    print("\n🔨 Building Agent Graph (Pydantic version)...")

    graph = StateGraph(AgentState)

    graph.add_node("reasoning", create_reasoning_node(llm))
    graph.add_node("tool_executor", tool_executor_node)
    graph.add_node("final_answer", final_answer_node)

    graph.add_edge(START, "reasoning")

    graph.add_conditional_edges(
        "reasoning",
        should_use_tool,
        {"tool_executor": "tool_executor", "final_answer": "final_answer"}
    )

    graph.add_conditional_edges(
        "tool_executor",
        after_tool,
        {"reasoning": "reasoning", "final_answer": "final_answer"}
    )

    graph.add_edge("final_answer", END)

    print("✅ Agent graph built!")
    return graph.compile()


# =============================================================================
# 8. MAIN
# =============================================================================

def run_agent(app, query: str) -> str:
    print(f"\n{'#'*60}")
    print(f"# QUERY: {query}")
    print(f"{'#'*60}")

    initial_state = {
        "query": query,
        "tool_decision": None,
        "tool_result": "",
        "iteration": 0,
        "final_answer": "",
        "history": []
    }

    final_state = app.invoke(initial_state)
    return final_state["final_answer"]


def main():
    print("=" * 60)
    print("  ReAct Agent with Pydantic Parsing")
    print("=" * 60)

    llm = create_llm()
    app = build_agent_graph(llm)

    # Show what the parser generates
    print("\n📋 Format instructions (sent to LLM):")
    print("-" * 40)
    print(parser.get_format_instructions()[:300] + "...")
    print("-" * 40)

    test_queries = [
        "What is LangChain?",
        "Calculate 25 * 4 + 10",
        "Tell me about Python",
    ]

    for query in test_queries:
        response = run_agent(app, query)
        print(response)

    # Interactive mode
    print("\n" + "=" * 60)
    print("  Interactive Mode (type 'quit' to exit)")
    print("=" * 60)

    while True:
        user_query = input("\n🔹 Enter your query: ").strip()
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        if user_query:
            print(run_agent(app, user_query))


if __name__ == "__main__":
    main()