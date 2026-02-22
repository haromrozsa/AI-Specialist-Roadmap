
# LangGraph: Query Classifier & Router

# Session Summary: A graph-based workflow using LangGraph that classifies user queries and routes them to specialized handlers based on query type (factual, creative, or code).


## What I Did During the Session

1. **Learned LangGraph Fundamentals**:
   - Understood core concepts: State (shared `TypedDict`), Nodes (processing functions), Edges (connections), and conditional routing
   - Compared LangChain (linear pipelines) vs LangGraph (flexible graphs with branches and loops)

2. **Built a Multi-Node Classification Workflow**:
   - Created `QueryState` to hold data flowing through the graph (`query`, `category`, `handler_response`, `final_response`)
   - Implemented 5 nodes: classifier, three specialized handlers (factual, creative, code), and response formatter
   - Each handler uses its own optimized `PromptTemplate` for its purpose

3. **Implemented Conditional Routing**:
   - Created `route_by_category()` function returning the next node name based on classification
   - Used `add_conditional_edges()` to dynamically route queries to appropriate handlers
   - All handlers converge back to a single formatter node before `END`

4. **Constructed and Compiled the StateGraph**:
   - Used `StateGraph(QueryState)` with `add_node()`, `add_edge()`, and `add_conditional_edges()`
   - Connected `START` → classify → handlers → formatter → `END`
   - Compiled with `graph.compile()` and executed with `app.invoke(initial_state)`

5. **Integrated LangChain Components Inside LangGraph Nodes**:
   - Reused `HuggingFacePipeline` LLM setup from previous work
   - Built LCEL chains (`prompt | llm | StrOutputParser()`) inside each node function

---

## What I Learned

1. **LangGraph Core Concepts**:
   - **State**: A `TypedDict` that flows through the graph; nodes return partial updates (only changed fields)
   - **Nodes**: Functions that receive state and return updates; can use LLM or pure Python logic
   - **Edges**: Fixed (`add_edge`) or conditional (`add_conditional_edges`) connections between nodes
   - **StateGraph**: Must be compiled before use; executed with `invoke()`

2. **LangChain vs LangGraph**:
   - **LangChain (LCEL)**: Linear pipelines, data transforms as it flows, no loops
   - **LangGraph**: Arbitrary graphs with branches, cycles, and persistent shared state
   - They work together: LangGraph orchestrates, LangChain components run inside nodes

3. **Conditional Routing Pattern**:
   - Routing functions return node names as strings (use `Literal` type hints)
   - `add_conditional_edges(source, router_func, mapping_dict)` enables dynamic paths
   - Allows building decision trees and complex workflows

4.**Graph Construction Pattern**:
   - Create → Add nodes → Add edges (START, conditional, END) → Compile → Invoke
   - Initial state only needs input fields; empty strings for fields nodes will fill
   - Final state contains all accumulated data from the entire workflow

---

This session provided hands-on experience building stateful, graph-based AI workflows with LangGraph, understanding how it extends LangChain's capabilities for complex routing and multi-step processing scenarios.