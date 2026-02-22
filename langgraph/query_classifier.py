"""
LangGraph Tutorial: Query Classifier & Router
=============================================
This example demonstrates:
1. State management with TypedDict
2. Multiple nodes with different responsibilities
3. Conditional routing based on classification
4. Integration with LangChain components (LLM, Prompts)
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# =============================================================================
# 1. STATE DEFINITION
# =============================================================================

class QueryState(TypedDict):
    """
    The shared state that flows through the graph.
    Each node can read from and write to this state.
    """
    query: str                    # Original user query
    category: str                 # Classification result: "factual", "creative", or "code"
    handler_response: str         # Response from the specialized handler
    final_response: str           # Formatted final response
    metadata: dict                # Additional info (for debugging/logging)


# =============================================================================
# 2. LLM SETUP
# =============================================================================

def create_llm(model_id: str = "microsoft/Phi-3-mini-4k-instruct"):
    """
    Create a HuggingFace LLM pipeline.
    Reusing the same pattern from your existing code.
    """
    print(f"🔄 Loading LLM: {model_id}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto"
    )

    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.7,
        top_k=50,
        do_sample=True,
        return_full_text=False
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    print(f"✅ LLM loaded successfully")
    return llm


# =============================================================================
# 3. NODE DEFINITIONS
# =============================================================================

def create_classify_node(llm):
    """
    Creates the classification node.
    This node determines the category of the user's query.
    """
    
    classify_prompt = PromptTemplate(
        template="""Classify the following query into exactly one category.
Categories:
- "factual": Questions seeking facts, definitions, explanations, or information
- "creative": Requests for stories, poems, creative writing, brainstorming
- "code": Questions about programming, code examples, debugging, or technical implementation

Query: {query}

Respond with ONLY one word (factual, creative, or code):""",
        input_variables=["query"]
    )
    
    classify_chain = classify_prompt | llm | StrOutputParser()
    
    def classify_node(state: QueryState) -> dict:
        """
        Classifies the query and returns the category.
        """
        print(f"\n📋 CLASSIFY NODE")
        print(f"   Input query: {state['query']}")
        
        result = classify_chain.invoke({"query": state["query"]})
        
        # Parse the result - extract the category keyword
        result_lower = result.lower().strip()
        
        if "creative" in result_lower:
            category = "creative"
        elif "code" in result_lower:
            category = "code"
        else:
            category = "factual"  # Default fallback
        
        print(f"   Classified as: {category}")
        
        return {
            "category": category,
            "metadata": {"raw_classification": result}
        }
    
    return classify_node


def create_factual_handler_node(llm):
    """
    Creates the factual question handler node.
    Optimized for accurate, informative responses.
    """
    
    factual_prompt = PromptTemplate(
        template="""You are a knowledgeable assistant focused on providing accurate, factual information.
Answer the following question clearly and concisely. Include relevant details but stay focused.

Question: {query}

Factual Answer:""",
        input_variables=["query"]
    )
    
    factual_chain = factual_prompt | llm | StrOutputParser()
    
    def factual_handler(state: QueryState) -> dict:
        """
        Handles factual queries with emphasis on accuracy.
        """
        print(f"\n📚 FACTUAL HANDLER NODE")
        
        response = factual_chain.invoke({"query": state["query"]})
        
        print(f"   Generated factual response")
        
        return {"handler_response": response}
    
    return factual_handler


def create_creative_handler_node(llm):
    """
    Creates the creative request handler node.
    Optimized for imaginative, engaging responses.
    """
    
    creative_prompt = PromptTemplate(
        template="""You are a creative assistant with a flair for imagination and storytelling.
Respond to the following request with creativity, originality, and engaging content.

Request: {query}

Creative Response:""",
        input_variables=["query"]
    )
    
    creative_chain = creative_prompt | llm | StrOutputParser()
    
    def creative_handler(state: QueryState) -> dict:
        """
        Handles creative requests with emphasis on imagination.
        """
        print(f"\n🎨 CREATIVE HANDLER NODE")
        
        response = creative_chain.invoke({"query": state["query"]})
        
        print(f"   Generated creative response")
        
        return {"handler_response": response}
    
    return creative_handler


def create_code_handler_node(llm):
    """
    Creates the code question handler node.
    Optimized for technical, programming-related responses.
    """
    
    code_prompt = PromptTemplate(
        template="""You are an expert programming assistant.
Provide a clear, well-commented code solution or technical explanation for the following query.
Use proper code formatting and explain your approach.

Query: {query}

Technical Response:""",
        input_variables=["query"]
    )
    
    code_chain = code_prompt | llm | StrOutputParser()
    
    def code_handler(state: QueryState) -> dict:
        """
        Handles code-related queries with emphasis on technical accuracy.
        """
        print(f"\n💻 CODE HANDLER NODE")
        
        response = code_chain.invoke({"query": state["query"]})
        
        print(f"   Generated code response")
        
        return {"handler_response": response}
    
    return code_handler


def response_formatter_node(state: QueryState) -> dict:
    """
    Final node that formats the response for output.
    This node doesn't need the LLM - it just structures the output.
    """
    print(f"\n📝 RESPONSE FORMATTER NODE")
    
    category_emoji = {
        "factual": "📚",
        "creative": "🎨",
        "code": "💻"
    }
    
    emoji = category_emoji.get(state["category"], "💬")
    
    final_response = f"""
{'='*60}
{emoji} Response Type: {state['category'].upper()}
{'='*60}

{state['handler_response']}

{'='*60}
"""
    
    print(f"   Formatted final response")
    
    return {"final_response": final_response}


# =============================================================================
# 4. ROUTING FUNCTION
# =============================================================================

def route_by_category(state: QueryState) -> Literal["factual_handler", "creative_handler", "code_handler"]:
    """
    Routing function for conditional edges.
    Returns the name of the next node based on the category.
    """
    category = state["category"]
    
    if category == "creative":
        return "creative_handler"
    elif category == "code":
        return "code_handler"
    else:
        return "factual_handler"


# =============================================================================
# 5. GRAPH CONSTRUCTION
# =============================================================================

def build_query_router_graph(llm):
    """
    Builds the complete LangGraph workflow.
    """
    print("\n🔨 Building LangGraph workflow...")
    
    # Create the graph with our state schema
    graph = StateGraph(QueryState)
    
    # -------------------------------------------------------------------------
    # Add nodes
    # -------------------------------------------------------------------------
    graph.add_node("classify", create_classify_node(llm))
    graph.add_node("factual_handler", create_factual_handler_node(llm))
    graph.add_node("creative_handler", create_creative_handler_node(llm))
    graph.add_node("code_handler", create_code_handler_node(llm))
    graph.add_node("response_formatter", response_formatter_node)
    
    # -------------------------------------------------------------------------
    # Add edges
    # -------------------------------------------------------------------------
    
    # START -> classify (entry point)
    graph.add_edge(START, "classify")
    
    # classify -> conditional routing to handlers
    graph.add_conditional_edges(
        "classify",              # Source node
        route_by_category,       # Routing function
        {                        # Mapping of return values to node names
            "factual_handler": "factual_handler",
            "creative_handler": "creative_handler",
            "code_handler": "code_handler"
        }
    )
    
    # All handlers -> response_formatter
    graph.add_edge("factual_handler", "response_formatter")
    graph.add_edge("creative_handler", "response_formatter")
    graph.add_edge("code_handler", "response_formatter")
    
    # response_formatter -> END
    graph.add_edge("response_formatter", END)
    
    # -------------------------------------------------------------------------
    # Compile the graph
    # -------------------------------------------------------------------------
    compiled_graph = graph.compile()
    
    print("✅ Graph built successfully!")
    
    return compiled_graph


# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================

def run_query(app, query: str) -> str:
    """
    Run a single query through the graph.
    """
    print(f"\n{'#'*60}")
    print(f"# NEW QUERY: {query}")
    print(f"{'#'*60}")
    
    # Initial state - only query is required, others will be filled by nodes
    initial_state = {
        "query": query,
        "category": "",
        "handler_response": "",
        "final_response": "",
        "metadata": {}
    }
    
    # Invoke the graph
    final_state = app.invoke(initial_state)
    
    return final_state["final_response"]


def main():
    """
    Main entry point - demonstrates the Query Router workflow.
    """
    print("=" * 60)
    print("  LangGraph Tutorial: Query Classifier & Router")
    print("=" * 60)
    
    # Step 1: Create the LLM
    llm = create_llm()
    
    # Step 2: Build the graph
    app = build_query_router_graph(llm)
    
    # Step 3: Test with different query types
    test_queries = [
        # Factual queries
        "What is the capital of France?",
        "Explain how photosynthesis works.",
        
        # Creative queries
        "Write a short poem about a robot learning to love.",
        "Give me a creative name for a coffee shop in space.",
        
        # Code queries
        "How do I read a CSV file in Python?",
        "Write a function to reverse a string in JavaScript.",
    ]
    
    print("\n" + "=" * 60)
    print("  Running Test Queries")
    print("=" * 60)
    
    for query in test_queries:
        response = run_query(app, query)
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
            response = run_query(app, user_query)
            print(response)


if __name__ == "__main__":
    main()
