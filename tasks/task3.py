# Task 3: Automatic Chain Evaluator & Cost Ledger
from langchain.schema import BaseRetriever
from typing import List
from langchain.schema import Document
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import re

# Mock QA pairs for Apple's 10-K (would normally load from CSV)
golden_qa_pairs = [
    {
        "question": "What are Apple's primary product categories?",
        "answer": "iPhone, Mac, iPad, Wearables, Services"
    },
    {
        "question": "What is Apple's gross margin percentage?",
        "answer": "approximately 43 percent"
    },
    {
        "question": "Where are Apple's main manufacturing partners located?",
        "answer": "China, Taiwan, other parts of Asia"
    },
    {
        "question": "How many retail stores does Apple operate?",
        "answer": "over 500 retail stores"
    },
    {
        "question": "What is Apple's approach to research and development?",
        "answer": "significant ongoing investment in R&D"
    }
]

class MockRetriever(BaseRetriever):
    """Mock retriever that returns consistent docs for testing"""
    def get_relevant_documents(self, query: str) -> List[Document]:
        # Return mock documents based on query
        if "product categories" in query.lower():
            return [Document(page_content="Apple's primary products include iPhone, Mac, iPad, Wearables and Services.")]
        elif "gross margin" in query.lower():
            return [Document(page_content="Apple's gross margin was approximately 43 percent last year.")]
        elif "manufacturing partners" in query.lower():
            return [Document(page_content="Apple's manufacturing is concentrated in China, Taiwan and other parts of Asia.")]
        elif "retail stores" in query.lower():
            return [Document(page_content="Apple operates over 500 retail stores worldwide.")]
        elif "research and development" in query.lower():
            return [Document(page_content="Apple continues to make significant ongoing investment in research and development.")]
        else:
            return [Document(page_content="Information not found in document.")]

class MockLLM:
    """Mock LLM that returns consistent answers for testing"""
    def invoke(self, prompt: str):
        if "product categories" in prompt:
            return "Apple's main products are iPhone, Mac, iPad, Wearables, and Services."
        elif "gross margin" in prompt:
            return "Apple reported a gross margin of approximately 43 percent."
        elif "manufacturing partners" in prompt:
            return "Apple's manufacturing is mainly in China, Taiwan, and other Asian countries."
        elif "retail stores" in prompt:
            return "Apple has over 500 retail stores globally."
        elif "research and development" in prompt:
            return "Apple invests significantly in ongoing research and development."
        else:
            return "I don't know the answer to that question."

class QAEvaluator:
    def __init__(self, qa_pairs, retriever, llm):
        self.qa_pairs = qa_pairs
        self.retriever = retriever
        self.llm = llm
        self.results = []
        self.total_cost = 0.0  # In dollars
    
    def normalize_text(self, text):
        """Normalize text for comparison"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()
    
    def calculate_f1(self, pred, true):
        """Calculate F1 score between predicted and true answer"""
        pred_tokens = set(self.normalize_text(pred).split())
        true_tokens = set(self.normalize_text(true).split())
        
        if not pred_tokens or not true_tokens:
            return 0.0
        
        common_tokens = pred_tokens & true_tokens
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(true_tokens)
        
        if (precision + recall) == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def mock_usage_metrics(self, text):
        """Mock function to simulate token usage tracking"""
        # In a real implementation, we'd use response.usage
        words = len(text.split())
        prompt_tokens = words + 50  # Base prompt size
        completion_tokens = words
        
        # Mock pricing: $0.0015 per 1K prompt tokens, $0.002 per 1K completion tokens
        prompt_cost = (prompt_tokens / 1000) * 0.0015
        completion_cost = (completion_tokens / 1000) * 0.002
        
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_cost": prompt_cost + completion_cost
        }
    
    def evaluate(self):
        """Evaluate the QA chain against golden pairs"""
        for pair in self.qa_pairs:
            # Simulate running through QA chain
            docs = self.retriever.get_relevant_documents(pair["question"])
            context = " ".join([doc.page_content for doc in docs])
            prompt = f"Context: {context}\nQuestion: {pair['question']}\nAnswer:"
            
            # Get LLM response
            response = self.llm.invoke(prompt)
            
            # Calculate metrics
            f1 = self.calculate_f1(response, pair["answer"])
            usage = self.mock_usage_metrics(prompt + response)
            cost_cents = usage["total_cost"] * 100  # Convert to cents
            
            self.total_cost += usage["total_cost"]
            
            self.results.append({
                "Question": pair["question"],
                "F1 Score": f1,
                "Cost (cents)": cost_cents
            })
    
    def get_results(self):
        """Return evaluation results"""
        return pd.DataFrame(self.results)
    
    def assert_metrics(self):
        """Check if mean F1 >= 0.6 and total cost <= $0.10"""
        mean_f1 = np.mean([r["F1 Score"] for r in self.results])
        if mean_f1 < 0.6:
            raise AssertionError(f"Mean F1 score {mean_f1:.2f} is below 0.6 threshold")
        
        if self.total_cost > 0.10:
            raise AssertionError(f"Total cost ${self.total_cost:.2f} exceeds $0.10 threshold")
        
        print(f"All metrics passed: Mean F1 = {mean_f1:.2f}, Total cost = ${self.total_cost:.2f}")

if __name__ == "__main__":
    print("\nStarting Task 3: Automatic Chain Evaluator & Cost Ledger")
    
    # Initialize components
    retriever = MockRetriever()
    llm = MockLLM()
    evaluator = QAEvaluator(golden_qa_pairs, retriever, llm)
    
    # Run evaluation
    evaluator.evaluate()
    results_df = evaluator.get_results()
    
    # Print results
    print("\nEvaluation Results:")
    print(results_df.to_string(index=False))
    
    # Check assertions
    try:
        evaluator.assert_metrics()
    except AssertionError as e:
        print(f"\nAssertion Error: {str(e)}")
    
    # Save results
    results_df.to_csv("evaluation_results.csv", index=False)
    print("\nResults saved to evaluation_results.csv")
    
    # Reflection
    print("\nTask 3 Reflection:")
    print("""
    Why I chose this method or approach:
    - F1 score provides balanced evaluation of answer quality
    - Mock components allow testing without real API calls
    - Cost tracking is essential for production systems
    
    What surprised me during development:
    - How sensitive F1 score is to phrasing differences
    - The importance of text normalization in evaluation
    - How quickly costs can accumulate with many queries
    
    What my next steps would be given additional time:
    - Implement real token counting with API responses
    - Add more sophisticated evaluation metrics (e.g., BLEU, ROUGE)
    - Create a visual dashboard for tracking metrics over time
    """)