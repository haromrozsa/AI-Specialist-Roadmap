# AI-Specialist-Roadmap

> 🚀 A comprehensive learning repository documenting my journey to becoming an AI specialist

## 📌 About This Repository

This repository serves as both a **learning project** and a **professional portfolio** demonstrating my commitment to mastering Artificial Intelligence. It showcases hands-on projects, experiments, and implementations across various AI/ML domains.

**Why this exists:**
- To systematically learn and document AI concepts, frameworks, and best practices
- To provide tangible evidence of my skills and continuous growth in the AI field
- To demonstrate that I invest time and effort into staying current with AI technologies

> ⚠️ **Note**: This repository represents only a portion of my AI/ML experience. I have been actively working with AI technologies since **2017**, including private and proprietary ML/AI projects that cannot be shared publicly.

## 🎓 Certifications & Education

| Certification/Education                                    | Issuer   | Year |
|------------------------------------------------------------|----------|------|
| **NVIDIA-Certified Associate: Generative AI LLMs**         | NVIDIA   | 2024 |
| **Artificial Intelligence for Trading Nanodegree Program** | Udacity  | 2018 |
| **Artificial Intelligence Nanodegree Program**             | Udacity  | 2017 |

## 👨‍💻 Background & Experience

- **7+ years** of hands-on experience with AI/ML technologies (since 2017)
- Developed **private ML/AI projects** as personal initiatives and side projects
- Self-driven learner staying current with the rapidly evolving AI landscape
- Practical experience spanning from classical ML to modern Generative AI
- Passion for AI that extends beyond work – continuous experimentation and learning
- 
## 🎯 Goals

- **Continuous Learning**: Push meaningful updates weekly to maintain learning momentum
- **Practical Experience**: Build real-world projects, not just theoretical knowledge
- **Portfolio Development**: Create a body of work that demonstrates senior-level AI capabilities
- **Knowledge Sharing**: Document learnings in a way that others can benefit from

## 📚 Topics Covered

| Category | Topics |
|----------|--------|
| **AI Fundamentals** | Neural Networks, Deep Learning, Model Architecture |
| **Machine Learning** | Supervised/Unsupervised Learning, Model Training & Evaluation |
| **NLP** | Hugging Face Transformers, Sentiment Analysis, Text Generation |
| **Generative AI** | LLMs, Prompt Engineering, Model Fine-tuning |
| **Frameworks** | PyTorch, TensorFlow, Keras, scikit-learn |
| **MLOps** | Model Deployment, Pipeline Development, Performance Optimization |
| **Model Training** | Fine-tuning, Transfer Learning, Frozen Base Training, Training from Scratch |
| **Model Usage** | Pre-trained Models, Fine-tuning, Inference Optimization |
| **Model Serving** | FastAPI, ONNX Runtime, REST API Inference |
| **Production APIs** | Pydantic Validation, Logging, Error Handling, Latency Measurement |
| **LLM Frameworks** | LangChain, Prompt Templates, Chain Composition |
| **Graph Workflows** | Stateful Processing, Conditional Routing, Multi-Node Orchestration |
| **AI Agents** | ReAct Pattern, Tool Usage, Autonomous Decision Making |
| **Vector Databases** | FAISS, Embeddings, Similarity Search |

## 🛠️ Projects & Implementations

### LangChain with Hugging Face
- Text generation chain implementing prompt → LLM → output pattern
- Integration of Hugging Face models with LangChain using `HuggingFacePipeline`
- Reusable `PromptTemplate` creation with dynamic input variables
- LangChain Expression Language (LCEL) for chain composition

### LangChain RAG (Retrieval-Augmented Generation)
- Complete RAG pipeline for grounded question answering from documents
- Document loading with `TextLoader`, `DirectoryLoader`, and `PyPDFLoader`
- Text chunking with `RecursiveCharacterTextSplitter` for optimal retrieval
- HuggingFace embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- FAISS vector store for efficient similarity search
- RAG chain composition using LCEL with `RunnablePassthrough` and `StrOutputParser`
- Source attribution to verify answers are grounded in documents

### LangGraph Query Classifier & Router
- Graph-based workflow for intelligent query routing
- State management with `TypedDict` for data flowing through nodes
- Multi-node architecture: classifier, specialized handlers (factual, creative, code), formatter
- Conditional routing using `add_conditional_edges()` for dynamic path selection
- Integration of LangChain components (prompts, LLM, parsers) inside LangGraph nodes
- Demonstrates LangGraph vs LangChain: flexible graphs with branches vs linear pipelines

### ReAct Agent with Tools
- Implementation of ReAct (Reasoning + Acting) pattern for autonomous agents
- Tool definition and execution (search, calculator) with agent-driven selection
- Three implementation approaches demonstrating abstraction levels:
  - **Manual parsing**: Regex-based extraction of tool decisions from LLM output
  - **Pydantic parsing**: Structured JSON output with `PydanticOutputParser` for type-safe validation
  - **LangChain built-in**: `@tool` decorator + `create_react_agent()` + `AgentExecutor`
- Conditional looping in LangGraph: reasoning → tool execution → reasoning (until done)
- Demonstrates how frameworks abstract parsing, routing, and loop management

### FastAPI ONNX Inference API
- End-to-end ML inference pipeline: train → export → serve
- Model training with scikit-learn `LogisticRegression` on digits dataset (64 features, 10 classes)
- ONNX export using `skl2onnx` with `FloatTensorType` for flexible batch sizes
- FastAPI `/predict` endpoint with `onnxruntime.InferenceSession` for inference
- Input validation, normalization, and proper HTTP error handling

### FastAPI ONNX Inference API
- End-to-end ML inference pipeline: train → export → serve
- Model training with scikit-learn `LogisticRegression` on digits dataset (64 features, 10 classes)
- ONNX export using `skl2onnx` with `FloatTensorType` for flexible batch sizes
- FastAPI `/predict` endpoint with `onnxruntime.InferenceSession` for inference
- Input validation, normalization, and proper HTTP error handling
- Structured logging with dual handlers (console + file) and UTF-8 encoding
- Custom exception handlers for validation errors and global error catching
- HTTP middleware for request/response latency measurement
- Proper handling of `skl2onnx` multi-output models (labels + probabilities)
- Comprehensive test suite covering valid requests and edge cases


### Hugging Face Pipelines
- Basic sentiment analysis pipeline using DistilBERT
- Model comparison tool benchmarking different NLP models
- Performance analysis and tradeoff evaluation

### Model Fine-tuning & Training Approaches
- Fine-tuning DistilBERT for sentiment analysis on IMDB dataset
- Comparison of three training approaches:
  - **Fine-tuning**: Training all layers with pre-trained weights
  - **Frozen Base**: Training only classification head (~1.5K vs ~66M parameters)
  - **From Scratch**: Random weight initialization with custom architecture
- Custom metrics implementation (accuracy, precision, recall, F1)
- Hugging Face Trainer API with TrainingArguments configuration

### Neural Networks with TensorFlow and PyTorch
- Feedforward neural networks implementation
- Training loops and evaluation metrics
- Regularization techniques (BatchNorm, Dropout, Max-norm)

*More projects being added continuously...*

## 🛠️ Development Approach

- Leveraging modern AI-assisted development tools (LLM-based coding assistants)
- Focus on understanding, reviewing, and optimizing AI-generated code
- Combining AI productivity with solid fundamentals

## 💡 Skills Demonstrated

- **Model Development**: Building, training, and optimizing neural networks
- **Model Fine-tuning**: Transfer learning, frozen base training, training from scratch
- **Model Export**: ONNX conversion for portable, framework-agnostic deployment
- **API Development**: FastAPI endpoints for ML model serving with proper validation
- **Production APIs**: Pydantic validation, structured logging, error handling, latency monitoring
- **Framework Proficiency**: Hands-on experience with PyTorch, TensorFlow, and Hugging Face
- **Hugging Face Expertise**: Trainer API, AutoModel, AutoTokenizer, DataCollators
- **LangChain Development**: Chain composition, prompt templates, LLM integration
- **LangGraph Workflows**: Stateful graphs, conditional routing, multi-node orchestration
- **AI Agents**: ReAct pattern implementation, tool integration, autonomous reasoning loops
- **RAG Systems**: Document ingestion, embeddings, vector stores, retrieval-augmented generation
- **Vector Databases**: FAISS integration, similarity search, embedding persistence
- **Generative AI**: NVIDIA-certified expertise in LLMs and generative models
- **Problem Solving**: Debugging ML pipelines and resolving compatibility issues
- **Best Practices**: Implementing proper regularization, evaluation, and documentation
- **Continuous Improvement**: Regular commits showing ongoing learning and development

## 📈 Progress Tracking

This repository is actively maintained with weekly updates. Each session is documented with:
- What was accomplished
- Key learnings and insights
- Code implementations and experiments

## 🤝 Connect

Feel free to explore the code, raise issues, or reach out if you'd like to discuss AI/ML topics!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*This repository reflects my dedication to becoming a proficient AI specialist through consistent practice and hands-on learning. It complements years of private experience and formal certifications in the AI field.*