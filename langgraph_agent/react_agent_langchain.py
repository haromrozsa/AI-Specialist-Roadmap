"""
LangGraph Tutorial: ReAct Agent using LangChain's Built-in Agent
================================================================
This example demonstrates:
1. Tool definition with @tool decorator
2. LangChain's create_react_agent() - automatic parsing & routing
3. AgentExecutor - handles the loop automatically
"""

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool


# =============================================================================
# 1. TOOLS DEFINITION - Just use @tool decorator!
# =============================================================================

@tool
def search(query: str) -> str:
    """Search for factual information about a topic. Use this for questions about facts, definitions, or general knowledge."""

    # Mock knowledge base
    knowledge = {
        "python": "Python is a high-level programming language created by Guido van Rossum in 1991.",
        "langchain": "LangChain is a framework for building applications with large language models.",
        "langgraph": "LangGraph is a library for building stateful, multi-agent applications with LLMs.",
        "eiffel tower": "The Eiffel Tower is 330 meters tall and located in Paris, France.",
        "react": "ReAct is a prompting pattern combining Reasoning and Acting for LLM agents.",
    }

    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return value
    return f"No specific information found for: {query}"


@tool
def calculator(expression: str) -> str:
    """Calculate a math expression. Use this for arithmetic operations like addition, subtraction, multiplication, division."""

    try:
        import re
        # Clean expression - only allow safe characters
        clean_expr = re.sub(r'[^0-9+\-*/().\s]', '', expression)
        result = eval(clean_expr)
        return f"{result}"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


# List of tools - framework reads the docstrings automatically!
tools = [search, calculator]


# =============================================================================
# 2. REACT PROMPT TEMPLATE
# =============================================================================

# Standard ReAct prompt format that LangChain expects
REACT_PROMPT = PromptTemplate.from_template(
    """Answer the following questions as best you can. You have access to the following tools:
    
    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    Thought:{agent_scratchpad}"""
)


# =============================================================================
# 3. LLM SETUP
# =============================================================================

class MockReActLLM:
    """
    Mock LLM that simulates ReAct-style responses.
    Replace with real LLM for production.
    """

    def __init__(self):
        self.call_count = 0
        self.last_observation = None

    def invoke(self, prompt, **kwargs):
        self.call_count += 1
        prompt_str = str(prompt)
        prompt_lower = prompt_str.lower()

        # Check if we have an observation (tool result) to process
        if "Observation:" in prompt_str:
            # Extract the last observation
            parts = prompt_str.split("Observation:")
            if len(parts) > 1:
                self.last_observation = parts[-1].strip().split("\n")[0]
                # Return final answer after getting observation
                return f" I now know the final answer\nFinal Answer: {self.last_observation}"

        # Decide which tool to use based on the question
        if any(op in prompt_str for op in ['+', '-', '*', '/']) or "calculate" in prompt_lower or "what is 2" in prompt_lower:
            import re
            # Extract math expression
            numbers = re.findall(r'[\d+\-*/\s()]+', prompt_str)
            expr = ''.join(numbers).strip()
            if expr:
                return f" I need to calculate this math expression\nAction: calculator\nAction Input: {expr}"

        if any(word in prompt_lower for word in ["what is", "who is", "tell me about", "explain"]):
            # Extract the topic
            question = prompt_str.split("Question:")[-1].split("\n")[0].strip() if "Question:" in prompt_str else "unknown"
            return f" I need to search for information about this topic\nAction: search\nAction Input: {question}"

        return " I can answer this directly\nFinal Answer: I don't have specific information about this topic."

    def bind(self, **kwargs):
        """Mock bind method for compatibility."""
        return self

    @property
    def _llm_type(self):
        return "mock"


def create_llm():
    """
    Create the LLM.
    Using MockReActLLM for demonstration without model download.
    """
    print("🤖 Using MockReActLLM for demonstration")
    return MockReActLLM()

    # =========================================================================
    # Uncomment below to use a real HuggingFace LLM:
    # =========================================================================
    # from langchain_huggingface import HuggingFacePipeline
    # from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    #
    # model_id = "microsoft/phi-2"
    # print(f"🔄 Loading LLM: {model_id}...")
    #
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     device_map="auto",
    #     torch_dtype="auto"
    # )
    #
    # hf_pipeline = pipeline(
    #     "text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     max_new_tokens=256,
    #     temperature=0.1,
    #     do_sample=True,
    #     return_full_text=False
    # )
    #
    # print("✅ LLM loaded successfully")
    # return HuggingFacePipeline(pipeline=hf_pipeline)


# =============================================================================
# 4. AGENT CREATION - Framework handles everything!
# =============================================================================

def create_agent():
    """
    Create the ReAct agent using LangChain's built-in functions.

    The framework automatically:
    - Reads tool names and descriptions from @tool decorators
    - Parses LLM output to extract Action/Action Input
    - Calls the appropriate tool
    - Loops until Final Answer is found
    """
    print("\n🔨 Creating ReAct Agent...")

    llm = create_llm()

    # Create the agent - ONE LINE!
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=REACT_PROMPT
    )

    # Wrap in executor - handles the loop automatically
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,          # Shows reasoning steps
        max_iterations=5,      # Prevent infinite loops
        handle_parsing_errors=True
    )

    print("✅ Agent created successfully!")

    return agent_executor


# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================

def run_agent(agent_executor, query: str) -> str:
    """Run a query through the agent."""

    print(f"\n{'#'*60}")
    print(f"# QUERY: {query}")
    print(f"{'#'*60}\n")

    # Just invoke with input - framework handles everything!
    result = agent_executor.invoke({"input": query})

    formatted = f"""
{'='*60}
🤖 Agent Response (LangChain Built-in)
{'='*60}
Query: {query}

Answer: {result['output']}
{'='*60}
"""
    return formatted


def main():
    print("=" * 60)
    print("  ReAct Agent using LangChain's Built-in Agent")
    print("=" * 60)

    # Show tool info - framework reads this automatically
    print("\n📋 Registered Tools:")
    print("-" * 40)
    for t in tools:
        print(f"  • {t.name}: {t.description[:60]}...")
    print("-" * 40)

    # Create agent
    agent_executor = create_agent()

    # Test queries
    test_queries = [
        "What is LangChain?",
        "Calculate 25 * 4",
        "Tell me about Python",
    ]

    print("\n" + "=" * 60)
    print("  Running Test Queries")
    print("=" * 60)

    for query in test_queries:
        response = run_agent(agent_executor, query)
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
            response = run_agent(agent_executor, user_query)
            print(response)


if __name__ == "__main__":
    main()