# Task 1: 10-K Retrieval QA

## Overview
This script implements a Retrieval-Augmented Question Answering (QA) system to extract information from SEC 10-K filings for a predefined set of companies. It downloads the latest 10-K filings, processes them into chunks, creates a vector store using FAISS, and answers two specific questions about each company: their primary revenue sources and their biggest supply chain risk. Results are saved to a CSV file (10k_qa_results.csv).

## Features
- Downloads 10-K filings using sec-edgar-downloader.
- Processes filings into text chunks using RecursiveCharacterTextSplitter.
- Embeds text using HuggingFaceEmbeddings (all-MiniLM-L6-v2).
- Uses FAISS for efficient vector-based retrieval.
- Answers questions using a RetrievalQA chain with an Ollama LLM (llama2).
- Includes fallback mock responses for robustness.
- Saves results incrementally to a CSV file.

## Prerequisites
- Python 3.8+
- Required packages: sec-edgar-downloader, langchain, langchain-community, langchain-huggingface, faiss-cpu, pandas, numpy
- Ollama installed with the llama2 model
- Internet access for downloading filings

## Installation
1. Clone the repository or copy the script.
2. Install dependencies:
   pip install sec-edgar-downloader langchain langchain-community langchain-huggingface faiss-cpu pandas numpy
3. Install and set up Ollama with the llama2 model:
   ollama pull llama2

## Usage
1. Ensure the sec_filings directory is writable.
2. Run the script:
   python task1.py
3. Check the output CSV file (10k_qa_results.csv) for results.

## Output
The script generates a CSV file (10k_qa_results.csv) with the following columns:
- Ticker: Company ticker symbol
- Company: Company name
- Sector: Company sector
- Revenue_Question: Question about revenue sources
- Revenue_Answer: Answer to the revenue question
- Risk_Question: Question about supply chain risk
- Risk_Answer: Answer to the risk question
- Status: Success or Failed

## Reflection
Why I chose this method or approach:
I chose a retrieval-augmented QA approach because 10-K filings are lengthy and unstructured, making direct LLM processing inefficient. Using FAISS for vector search and RecursiveCharacterTextSplitter for chunking allowed efficient retrieval of relevant text segments. The Ollama LLM was selected for its open-source nature, and mock responses were included to ensure robustness against failures in downloading or processing filings.

What surprised me during development:
I was surprised by the variability in 10-K filing formats, which required robust text processing and error handling. The embeddings model (all-MiniLM-L6-v2) performed better than expected for financial text, but the LLM occasionally struggled with precise answers, necessitating careful prompt design.

What my next steps would be given additional time:
With more time, I would enhance the system by fine-tuning the LLM on financial texts to improve answer precision, implement more sophisticated text cleaning to handle filing inconsistencies, and add a user interface to allow interactive querying of the 10-K data.

# Task 2: LangGraph Financial Tool Router

## Overview
This script implements a financial tool router using LangGraph to process user queries related to stock prices, news headlines, and financial ratios. It routes queries to appropriate tools (price_lookup, news_headlines, stat_ratios) using an LLM, executes the selected tool, and formats the response. Results are saved to a text file (financial_tool_results.txt) in JSON format.

## Features
- Uses LangGraph to define a stateful workflow with three nodes: routing, tool execution, and response composition.
- Supports three tools: stock price lookup, news headlines, and financial ratios (P/E, P/S, ROE) using yfinance.
- Implements caching to reduce API calls and improve performance.
- Includes robust error handling and a mock LLM fallback for reliability.
- Saves query results with metadata (timestamp, status, tool used) to a file.

## Prerequisites
- Python 3.8+
- Required packages: langgraph, langchain-community, yfinance, requests, pandas
- Ollama installed with the llama2 model (optional, as mock LLM is available)
- Internet access for yfinance API calls

## Installation
1. Clone the repository or copy the script.
2. Install dependencies:
   pip install langgraph langchain-community yfinance requests pandas
3. (Optional) Install and set up Ollama with the llama2 model:
   ollama pull llama2

## Usage
1. Ensure the financial_data directory is writable.
2. Run the script to execute demo queries:
   python task2.py
3. Check the output file (financial_tool_results.txt) for results.

## Output
The script generates a text file (financial_tool_results.txt) with one JSON object per query, containing:
- query: The user query
- response: The formatted response
- status: Success or Failed
- tool_used: The selected tool (or "none")
- timestamp: ISO timestamp of the query

## Demo Queries
The script runs the following demo queries:
- "Give me the P/E ratio for NVDA"
- "What's the current price of Apple stock?"
- "Show me recent news about Tesla"
- "What are the financial ratios for Microsoft?"
- "Tell me about the weather" (demonstrates graceful failure)

## Reflection
Why I chose this method or approach:
I chose LangGraph for its ability to manage complex workflows with clear state transitions, making it ideal for routing and executing financial tools. The yfinance library was selected for its reliable access to financial data, and caching was implemented to optimize performance. The mock LLM fallback ensured the system could function even if the Ollama LLM failed.

What surprised me during development:
I was surprised by how often the LLM returned malformed JSON, requiring robust parsing logic. The yfinance API occasionally returned incomplete data, which highlighted the need for caching and error handling. The LangGraph framework was more intuitive to use than expected, simplifying workflow definition.

What my next steps would be given additional time:
Given more time, I would integrate additional financial tools (e.g., earnings reports), improve the LLM's JSON output consistency through prompt engineering or fine-tuning, and develop a web interface to allow real-time user queries with visualized outputs.


# Task 3: Automatic Chain Evaluator & Cost Ledger

## Overview
This script implements an evaluator for a Question Answering (QA) chain designed to answer questions about Apple's 10-K filing. It uses mock retriever and LLM components to simulate the QA process, evaluates answers against golden QA pairs using F1 score, and tracks mock API costs. Results are saved to a CSV file (evaluation_results.csv).

## Features
- Evaluates QA chain performance using F1 score for answer quality.
- Tracks mock API costs based on token usage (prompt and completion).
- Uses mock retriever and LLM for consistent testing without real API calls.
- Normalizes text for fair comparison between predicted and true answers.
- Enforces quality (mean F1 >= 0.6) and cost (total <= $0.10) thresholds.
- Saves evaluation results to a CSV file.

## Prerequisites
- Python 3.8+
- Required packages: langchain, pandas, numpy, scikit-learn
- No external APIs or models required (uses mock components)

## Installation
1. Clone the repository or copy the script.
2. Install dependencies:
   pip install langchain pandas numpy scikit-learn

## Usage
1. Run the script:
   python task3.py
2. Check the output CSV file (evaluation_results.csv) for results.

## Output
The script generates a CSV file (evaluation_results.csv) with the following columns:
- Question: The evaluated question
- F1 Score: F1 score comparing predicted and true answers
- Cost (cents): Mock API cost for the query in cents

## Golden QA Pairs
The script evaluates the following questions (mocked for Apple's 10-K):
- What are Apple's primary product categories?
- What is Apple's gross margin percentage?
- Where are Apple's main manufacturing partners located?
- How many retail stores does Apple operate?
- What is Apple's approach to research and development?

## Reflection
Why I chose this method or approach:
I chose the F1 score for evaluation because it balances precision and recall, providing a robust measure of answer quality for financial texts. Mock components were used to ensure reproducible results without relying on external APIs. Cost tracking was included to simulate production considerations, and text normalization was critical for fair comparisons.

What surprised me during development:
I was surprised by how sensitive the F1 score was to minor phrasing differences, emphasizing the need for robust text normalization. The mock cost model revealed how quickly token-based costs can accumulate, even for small queries. The simplicity of implementing the evaluator was a pleasant surprise, thanks to the modular design.

What my next steps would be given additional time:
With more time, I would integrate real API token counting, add advanced evaluation metrics like BLEU or ROUGE, and create a dashboard to visualize evaluation results and cost trends over time. I would also expand the golden QA pairs to cover more diverse questions.