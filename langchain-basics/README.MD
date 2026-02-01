# LangChain with Hugging Face Training

# Session Summary: Building a Text Generation Chain with LangChain and Hugging Face

## What I Did During the Session

1. **Set Up LangChain with Hugging Face Integration**:
   - Installed the official `langchain-huggingface` partner package jointly maintained by Hugging Face and LangChain
   - Resolved various setup issues including Windows permission errors and circular import problems

2. **Built a Simple Text Generation Chain**:
   - Created `text_generation_chain.py` implementing the prompt → LLM → output pattern
   - Used `HuggingFacePipeline` wrapper to integrate Hugging Face models into LangChain
   - Configured a text generation pipeline with parameters like `max_new_tokens`, `temperature`, and `top_k`
   - Created a reusable `PromptTemplate` with input variables for dynamic prompt generation

3. **Implemented the Chain Pattern Using LCEL**:
   - Used LangChain Expression Language (LCEL) syntax: `chain = prompt | llm`
   - Executed the chain with `chain.invoke()` passing input variables as a dictionary
   - Tested with a question about Python benefits for data science

4. **Troubleshooted Common Setup Issues**:
   - Fixed circular import error caused by naming file same as package (`langchain_huggingface.py`)
   - Resolved pydantic version conflicts between LangChain (v2) and spaCy (v1)

## What I Learned

1. **LangChain Core Concepts**:
   - **LangChain** is a framework for building LLM-powered applications with modular, composable components
   - The basic pattern is: Input → Prompt Template → LLM → Output
   - **LCEL (LangChain Expression Language)** uses the `|` operator to chain components together
   - `chain.invoke()` executes the chain and passes variables through each component

2. **Hugging Face Integration with LangChain**:
   - `langchain-huggingface` is the official partner package for Hugging Face integration
   - `HuggingFacePipeline` wraps Hugging Face transformers pipelines for use in LangChain
   - Can load models locally using `AutoModelForCausalLM` and `AutoTokenizer`
   - `device_map="auto"` automatically uses GPU if available

3. **PromptTemplate Usage**:
   - `PromptTemplate` creates reusable templates with placeholders (e.g., `{question}`)
   - `input_variables` defines which placeholders the template expects
   - Templates help standardize prompts and make them parameterized for different inputs

4. **Text Generation Pipeline Parameters**:
   - `max_new_tokens`: Limits the length of generated text
   - `temperature`: Controls randomness (higher = more creative, lower = more deterministic)
   - `top_k`: Limits vocabulary choices to top K most likely tokens
   - `do_sample=True`: Enables sampling for diverse outputs (vs greedy decoding)

5. **Python Environment Best Practices**:
   - **Never name files the same as packages** you're importing (causes circular imports)
   - Solving dependency conflicts (like pydantic v1 vs v2)
   - Always verify which Python interpreter is active with `where python` or `which python`

This session provided hands-on experience with LangChain's core abstraction patterns and practical knowledge of setting up a working LLM pipeline with locally-run Hugging Face models.