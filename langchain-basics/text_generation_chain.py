from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 1. Set up the Hugging Face model and tokenizer
model_id = "microsoft/Phi-3-mini-4k-instruct"  # You can use other models like "gpt2", "distilgpt2", etc.

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",  # Automatically uses GPU if available
    torch_dtype="auto"
)

# 2. Create a Hugging Face pipeline for text generation
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    temperature=0.7,
    top_k=50,
    do_sample=True
)

# 3. Wrap the pipeline with LangChain's HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# 4. Create a PromptTemplate
template = """You are a helpful assistant. Answer the following question concisely.

Question: {question}

Answer:"""

prompt = PromptTemplate(
    template=template,
    input_variables=["question"]
)

# 5. Create and execute the chain (prompt → LLM → output)
chain = prompt | llm

# 6. Run the chain with an input
question = "What are the benefits of using Python for data science?"
response = chain.invoke({"question": question})

print("=" * 50)
print(f"Question: {question}")
print("=" * 50)
print(f"Response: {response}")