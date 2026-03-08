"""
LangGraph Tutorial: ReAct Agent with Tools
==========================================
This example demonstrates:
1. Tool definition and execution
2. Agent reasoning loop (ReAct pattern)
3. Conditional looping in LangGraph
4. LLM-driven decision making
"""

from typing import TypedDict, Literal, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re
import operator


# =============================================================================
# 1. TOOLS DEFINITION
# =============================================================================

def search_tool(query: str) -> str:
    """
    Simulated search tool - returns mock results.
    In production, this would call a real search API.
    """
    # Mock knowledge base
    knowledge = {
        "python": "Python is a high-level programming language created by Guido van Rossum in 1991.",
        "langchain": "LangChain is a framework for building applications with large language models.",
        "langgraph": "LangGraph is a library for building stateful, multi-agent applications with LLMs.",
        "eiffel tower": "The Eiffel Tower is 330 meters tall and located in Paris, France.",
        "default": "I found some information but it's not specific to your query."
    }

    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return value
    return knowledge["default"]


def calculator_tool(expression: str) -> str:
    """
    Simple calculator tool - evaluates math expressions safely.
    """
    try:
        # Clean the expression
        clean_expr = re.sub(r'[^0-9+\-*/().\s]', '', expression)
        result = eval(clean_expr)
        return f"The result of {clean_expr} is {result}"
    except Exception as e:
        return f"Could not calculate: {expression}. Error: {str(e)}"


TOOLS = {
    "search": {
        "function": search_tool,
        "description": "Search for factual information. Use for questions about facts, definitions, or general knowledge."
    },
    "calculator": {
        "function": calculator_tool,
        "description": "Calculate math expressions. Use for arithmetic, calculations, or number-related questions."
    }
}


# =============================================================================
# 2. STATE DEFINITION
# =============================================================================

class AgentState(TypedDict):
    """
    State for the ReAct agent.
    """
    query: str                              # Original user query
    reasoning: str                          # Agent's current reasoning
    selected_tool: str                      # Tool to use: "search", "calculator", or "none"
    tool_input: str                         # Input to pass to the tool
    tool_result: str                        # Result from tool execution
    iteration: int                          # Current iteration count
    final_answer: str                       # Final response to user
    history: Annotated[list, operator.add]  # Accumulating history of steps


# =============================================================================
# 3. LLM SETUP (Simplified - using mock for demo)
# =============================================================================

class MockLLM:
    """
    Mock LLM for demonstration without requiring model download.
    Replace with real LLM (HuggingFacePipeline) for production.
    """

    def invoke(self, prompt: str) -> str:
        prompt_lower = prompt.lower()

        # Simple pattern matching to simulate LLM reasoning
        if "calculate" in prompt_lower or any(op in prompt for op in ['+', '-', '*', '/']):
            if "what" in prompt_lower and any(char.isdigit() for char in prompt):
                # Extract math expression
                numbers = re.findall(r'[\d+\-*/\s()]+', prompt)
                expr = ''.join(numbers).strip()
                if expr:
                    return f"TOOL: calculator\nINPUT: {expr}\nREASON: This requires calculation."

        if any(word in prompt_lower for word in ["what is", "who is", "tell me about", "search"]):
            return f"TOOL: search\nINPUT: {prompt.split('Query:')[-1].strip()}\nREASON: This requires searching for information."

        if "tool_result" in prompt_lower or "answer the user" in prompt_lower:
            return "Based on the information gathered, I can now provide the answer."

        return "TOOL: none\nREASON: I can answer this directly without tools."


def create_llm():
    """
    Create the LLM. Using MockLLM for easy demonstration.
    Uncomment the HuggingFace code for real LLM.
    """
    print("🤖 Using MockLLM for demonstration")
    return MockLLM()

    # Uncomment below for real LLM:
    # from langchain_huggingface import HuggingFacePipeline
    # from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    #
    # model_id = "microsoft/phi-2"
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")
    # hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
    # return HuggingFacePipeline(pipeline=hf_pipeline)


# =============================================================================
# 4. NODE DEFINITIONS
# =============================================================================

def create_reasoning_node(llm):
    """
    The agent's reasoning node - decides what to do next.
    """

    def reasoning_node(state: AgentState) -> dict:
        print(f"\n🧠 REASONING NODE (iteration {state['iteration'] + 1})")

        # Build prompt based on current state
        if state["tool_result"]:
            # We have a tool result, decide if we need more info or can answer
            prompt = f"""You have received information from a tool.

Query: {state['query']}
Tool Result: {state['tool_result']}

Can you now answer the user's question? If yes, respond with:
TOOL: none
ANSWER: [your final answer]

If you need more information, specify another tool to use."""

        else:
            # Initial reasoning - decide which tool to use
            prompt = f"""You are a helpful assistant with access to these tools:

1. search - {TOOLS['search']['description']}
2. calculator - {TOOLS['calculator']['description']}

Query: {state['query']}

Decide which tool to use (or 'none' if you can answer directly).
Respond in this format:
TOOL: [search/calculator/none]
INPUT: [input for the tool]
REASON: [why you chose this]"""

        response = llm.invoke(prompt)
        print(f"   LLM Response: {response[:100]}...")

        # Parse the response
        tool = "none"
        tool_input = ""

        if "TOOL: search" in response:
            tool = "search"
            input_match = re.search(r'INPUT:\s*(.+?)(?:\n|REASON:|$)', response, re.IGNORECASE)
            tool_input = input_match.group(1).strip() if input_match else state["query"]
        elif "TOOL: calculator" in response:
            tool = "calculator"
            input_match = re.search(r'INPUT:\s*(.+?)(?:\n|REASON:|$)', response, re.IGNORECASE)
            tool_input = input_match.group(1).strip() if input_match else ""

        # Check if we have a final answer
        final_answer = ""
        if "ANSWER:" in response:
            answer_match = re.search(r'ANSWER:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
            final_answer = answer_match.group(1).strip() if answer_match else ""

        print(f"   Selected tool: {tool}")

        return {
            "reasoning": response,
            "selected_tool": tool,
            "tool_input": tool_input,
            "final_answer": final_answer,
            "iteration": state["iteration"] + 1,
            "history": [f"Iteration {state['iteration'] + 1}: Chose {tool}"]
        }

    return reasoning_node


def tool_executor_node(state: AgentState) -> dict:
    """
    Executes the selected tool and returns the result.
    """
    print(f"\n🔧 TOOL EXECUTOR NODE")

    tool_name = state["selected_tool"]
    tool_input = state["tool_input"]

    if tool_name in TOOLS:
        print(f"   Executing: {tool_name}({tool_input})")
        result = TOOLS[tool_name]["function"](tool_input)
        print(f"   Result: {result}")
    else:
        result = "No tool executed."

    return {
        "tool_result": result,
        "history": [f"Tool {tool_name} returned: {result[:50]}..."]
    }


def final_answer_node(state: AgentState) -> dict:
    """
    Formats the final answer for the user.
    """
    print(f"\n✅ FINAL ANSWER NODE")

    if state["final_answer"]:
        answer = state["final_answer"]
    elif state["tool_result"]:
        answer = f"Based on my research: {state['tool_result']}"
    else:
        answer = "I was unable to find an answer to your question."

    formatted = f"""
{'='*60}
🤖 Agent Response
{'='*60}
Query: {state['query']}

Answer: {answer}

Steps taken: {state['iteration']}
{'='*60}
"""

    print(f"   Generated final answer")

    return {"final_answer": formatted}


# =============================================================================
# 5. ROUTING FUNCTION
# =============================================================================

def should_use_tool(state: AgentState) -> Literal["tool_executor", "final_answer"]:
    """
    Decides whether to execute a tool or provide final answer.
    """
    # Safety: prevent infinite loops
    if state["iteration"] >= 3:
        print("   ⚠️ Max iterations reached, forcing final answer")
        return "final_answer"

    if state["selected_tool"] in ["search", "calculator"]:
        return "tool_executor"
    else:
        return "final_answer"


def after_tool(state: AgentState) -> Literal["reasoning", "final_answer"]:
    """
    After tool execution, decide whether to reason again or finish.
    """
    # If we have enough info or hit iteration limit, finish
    if state["iteration"] >= 3:
        return "final_answer"

    # Go back to reasoning to process the tool result
    return "reasoning"


# =============================================================================
# 6. GRAPH CONSTRUCTION
# =============================================================================

def build_agent_graph(llm):
    """
    Builds the ReAct agent graph with tool loop.
    """
    print("\n🔨 Building Agent Graph...")

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("reasoning", create_reasoning_node(llm))
    graph.add_node("tool_executor", tool_executor_node)
    graph.add_node("final_answer", final_answer_node)

    # Add edges
    graph.add_edge(START, "reasoning")

    # Conditional: after reasoning, either use tool or give final answer
    graph.add_conditional_edges(
        "reasoning",
        should_use_tool,
        {
            "tool_executor": "tool_executor",
            "final_answer": "final_answer"
        }
    )

    # After tool execution, go back to reasoning (loop!)
    graph.add_conditional_edges(
        "tool_executor",
        after_tool,
        {
            "reasoning": "reasoning",
            "final_answer": "final_answer"
        }
    )

    # Final answer ends the graph
    graph.add_edge("final_answer", END)

    compiled = graph.compile()
    print("✅ Agent graph built successfully!")

    return compiled


# =============================================================================
# 7. MAIN EXECUTION
# =============================================================================

def run_agent(app, query: str) -> str:
    """
    Run a query through the agent.
    """
    print(f"\n{'#'*60}")
    print(f"# QUERY: {query}")
    print(f"{'#'*60}")

    initial_state = {
        "query": query,
        "reasoning": "",
        "selected_tool": "",
        "tool_input": "",
        "tool_result": "",
        "iteration": 0,
        "final_answer": "",
        "history": []
    }

    final_state = app.invoke(initial_state)

    return final_state["final_answer"]


def main():
    print("=" * 60)
    print("  LangGraph Tutorial: ReAct Agent with Tools")
    print("=" * 60)

    # Create LLM and build graph
    llm = create_llm()
    app = build_agent_graph(llm)

    # Test queries
    test_queries = [
        "What is LangChain?",
        "Calculate 25 * 4 + 10",
        "Tell me about the Eiffel Tower",
        "What is 100 / 5?",
    ]

    print("\n" + "=" * 60)
    print("  Running Test Queries")
    print("=" * 60)

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
            response = run_agent(app, user_query)
            print(response)


if __name__ == "__main__":
    main()